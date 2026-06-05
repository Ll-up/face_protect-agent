import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import asyncio
from tool.mcp_tool import load_filesystem_tools

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from memory.memory_context import SessionManager
from tool.All import get_weather, location, request_human_approval
import logging
#日志配置
logging.basicConfig(
    level=logging.INFO,  # 级别：DEBUG < INFO < WARNING < ERROR
    # 日志格式：时间 + 日志级别 + 日志内容
    format="%(asctime)s - %(levelname)s - %(message)s",
    # 输出位置：1. 控制台 2. 日志文件（rag_logs.log）
    handlers=[
        #logging.FileHandler("rag_logs.log", encoding="utf-8"),  # 写入文件
        logging.StreamHandler()  # 输出到控制台
    ]
)
# 定义日志对象，后续用logger.info/error记录
logger = logging.getLogger(__name__)

sys_p='''
你是专业智能皮肤管理助手，具备两大核心能力：
1. 可调用工具函数获得位置和今日天气信息：紫外线强度、环境温度、空气湿度、天气状况（晴/阴/雨/大风等）；
2. 有本地知识库，可对用户输入的问题在知识库中提取信息，基于专业知识提供皮肤管理建议
你的任务：
当用户询问出行建议时，用工具函数获得当地今日天气信息，结合环境对皮肤的影响逻辑和皮肤管理专业知识，为用户生成当日专属「护肤方案+出行防护建议」。
当用户询问皮肤知识时，根据相关专业知识做出解答

严格遵循规则：
一、询问出行建议时
    1. 必须结合 紫外线、温度、湿度 三个核心指标做针对性分析；
    2. 区分高紫外线、干燥低湿度、高温、低温大风等不同环境对皮肤的伤害；
    3. 建议分模块输出：当日天气皮肤影响分析、晨间护肤要点、日间防晒防护、晚间修护护理、出行穿搭&防护小贴士；
    4. 语言通俗易懂，不夸大、不推销产品，只给科学护肤建议；
    5. 若用户未指定肤质，给出通用版建议；若后续有用户肤质信息，可自动适配对应肤质方案；
二、询问皮肤知识时
    必须结合皮肤管路的专业知识来回答用户问题

输出格式固定：
一、询问出行建议时
    【今日环境皮肤研判】
    结合紫外线/温度/湿度分析对皮肤的影响
    
    【晨间护肤建议】
    清洁、保湿、打底注意事项
    
    【日间防晒&防护建议】
    防晒等级、防晒选择、补涂时机、紫外线防护要点
    
    【晚间修护护理建议】
    清洁方式、保湿修护、舒缓重点
    
    【出行专属小贴士】
    穿搭、口罩帽子防护、室内外皮肤适应技巧
二、询问皮肤知识时
    【根据知识库中检索到的专业知识回答】
    ......
    
'''

load_dotenv()
BASE_URL=os.getenv("BASE_URL")
api_key=os.getenv("DASHSCOPE_API_KEY")


# 全局变量（在异步初始化后设置）
llm = None
llm_with_tools = None
tool_map = {}
TOOLS_REQUIRING_APPROVAL = {"location"}


async def init_tools():
    """异步初始化 MCP 工具和 LLM"""
    global llm, llm_with_tools, tool_map

    # 在异步函数内使用 await
    mcp_tools = await load_filesystem_tools(root_dir="./data")
    logger.info(f"加载了 {len(mcp_tools)} 个 MCP 文件系统工具")

    # 合并工具
    custom_tools = [get_weather, location]
    all_tools = mcp_tools + custom_tools

    # 初始化 LLM
    llm = ChatOpenAI(
        model="qwen3.5-plus",
        base_url=BASE_URL,
        api_key=api_key
    )
    llm_with_tools = llm.bind_tools(all_tools)

    # 构建工具映射
    tool_map = {t.name: t for t in all_tools}

    # 打印可用工具
    logger.info(f"可用工具: {list(tool_map.keys())}")

#加载向量库
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = FAISS.load_local(
    r"E:\Agent\face_protect-agent\faiss_index",          #文件夹名字
    embeddings,
    allow_dangerous_deserialization=True #自己的文件夹可信任
)
logging.info("向量库加载完成！")


#组装上下文 组装顺序：sys_prompt在最前 对话历史 用户输入
def build_context(
    user_input: str,
    history: list[dict],
    system: str = sys_p,
) -> list:
    """组装 LLM 可用的 messages：SystemMessage + 历史 + 当前 UserMessage。"""
    messages: list = []
    if system:
        messages.append(SystemMessage(content=system))
    for h in history:
        role, content = h.get("role", ""), h.get("content", "") #若无role就返回空字符串，安全读取 防止报错
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    #向量库中检索信息
    retrieved_docs = vectorstore.similarity_search(user_input, k=2)
    docs_content = "\n".join([doc.page_content for doc in retrieved_docs])
    if docs_content:
        logging.info(f"rag检索成功，信息为",{docs_content})
        knowledge_prompt = f"""
        以下是皮肤管理相关专业知识，请你基于这些知识回答用户问题：
        {docs_content}
        ------------------------------------------------
        用户问题：{user_input}
        """
        messages.append(HumanMessage(content=knowledge_prompt))
        logging.info(f"上下文组装完毕")
    else:
        logging.info(f"rag未检索到有效信息")
        knowledge_prompt = f"""
                未检索到相关片段，请仅基于常识与工具信息回答，勿编造专业细节
                ------------------------------------------------
                用户问题：{user_input}
                """
        messages.append(HumanMessage(content=knowledge_prompt))
        logging.info(f"上下文组装完毕")
    return messages

#记忆总结
def session_summerize(context: list[dict])->str:
    messages = []
    llm_0 = ChatOpenAI(model="qwen-flash", base_url=BASE_URL, api_key=api_key)
    sys='''你的任务是总结历史对话'''
    messages.append(SystemMessage(content=sys))
    res=llm_0.invoke(messages)
    return res

#模型运行主函数
def run_agent(user_input:str,
              session_id: str,
              context: list[dict],
              session_manager:SessionManager,
              max_steps: int = 3, #最大循环轮数
              )->str:
    messages = build_context(user_input,context)
    for step in range(max_steps):
        logging.info(f"正在进行第{step+1}轮推理循环")
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            reply = (response.content or "").strip() or "（无回复）"
            session_manager.append_history(session_id, "user", user_input)  # 存入记忆
            session_manager.append_history(session_id, "assistant", reply)
            return reply

        for call in response.tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            tid = call.get("id", "") #tid为工具调用的身份证
            if name not in tool_map:
                messages.append(ToolMessage(content=f"未知工具: {name}", tool_call_id=tid))
                continue #跳出该循环，返回信息给llm
            try:
                #HITL
                if name in TOOLS_REQUIRING_APPROVAL:
                    logging.info(f"进入HITL流程")
                    decision = request_human_approval(user_input, name, args)
                    if decision == "approve":
                        try:
                            result = tool_map[name].invoke(args)
                            messages.append(ToolMessage(content=str(result), tool_call_id=tid))
                        except Exception as e:
                            messages.append(ToolMessage(content=f"执行失败: {e}", tool_call_id=tid))
                    else:
                        reason = decision.replace("reject:", "").strip() if "reject:" in decision else "用户拒绝执行"
                        messages.append(
                            ToolMessage(
                                content=f"[人工拒绝] {reason}。请根据此反馈调整回复或结束，不要再次请求执行该操作。",
                                tool_call_id=tid,
                            )
                        )
                    continue
                else:
                    result = tool_map[name].invoke(args)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tid)) #tid将结果与ai想调用的工具对应
            except Exception as e:
                messages.append(ToolMessage(content=f"执行失败: {e}", tool_call_id=tid))
    session_manager.append_history(session_id, "user",user_input)
    session_manager.append_history(session_id, "assistant", "达到最大步数，未得到最终回复。")
    return "达到最大步数，未得到最终回复。"

def main() -> None:
    user_demo="01"
    Session = SessionManager()
    session_li=Session.create_session(user_demo)
    session_id=session_li["id"]
    print("输入问题，Agent 将自动选择工具并多轮调用后回答。输入 quit 退出。\n")
    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        contetx_history = Session.get_history(session_id)  # 历史对话
        reply = run_agent(user_input,session_id,contetx_history,Session)
        print("助手:", reply, "\n")

if __name__ == "__main__":
        main()
