import random
from langchain_core.tools import tool


@tool
def get_weather(city: str)->str:
    """
    当用户询问某地天气时，使用此工具查询该城市的天气。
    """
    back=f"{city}，多云 26°C-33°C 紫外线指数日均8 湿度62% 风力3级"
    return back

@tool
def location()->str:
    """获取用户所在城市的名称，以纯字符串形式返回,城市名可用于查询天气"""
    return random.choice(["深圳", "合肥", "杭州"])

def request_human_approval(goal: str, tool_name: str, args: dict) -> str:
    """
    审批型 HITL：呈现当前意图、待执行动作与参数、风险提示，等待人输入。
    返回 "approve" | "reject" 或 "reject:原因"
    """
    print("\n  ---------- [HITL 审批] ----------")
    print("  当前意图：", goal[:100] + ("..." if len(goal) > 100 else ""))
    print("  待执行动作：", tool_name)
    print("  参数：", args)
    print("  风险提示：涉及地点查询，请确认是否允许执行。")
    print("  输入 y 通过 / n 或 n:原因 拒绝 ----------")
    raw = input("  你的决策 (y/n): ").strip().lower()  #.lower()转小写
    if raw.startswith("y") or raw == "yes":
        return "approve"
    if raw.startswith("n") or raw == "no":
        if ":" in raw:
            return "reject:" + raw.split(":", 1)[1].strip()
        return "reject:用户拒绝执行"
    return "reject:无效输入，视为拒绝"