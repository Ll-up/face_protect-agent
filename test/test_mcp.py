# test_mcp.py
import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient


async def test_mcp():
    root_dir = "."

    # 确保目录存在并创建测试文件
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # 创建一个测试文件
    test_file = os.path.join(root_dir, "test.txt")
    if not os.path.exists(test_file):
        with open(test_file, "w") as f:
            f.write("Hello MCP! This is a test file.")

    print(f"服务器根目录: {root_dir}")

    client = MultiServerMCPClient({
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", root_dir],
        }
    })

    try:
        print("正在获取工具列表...")
        tools = await client.get_tools()
        print(f"✅ 成功加载 {len(tools)} 个工具")

        # 测试 list_directory
        print("\n--- 测试 list_directory 工具 ---")
        list_tool = next((t for t in tools if t.name == "list_directory"), None)

        if list_tool:
            # ✅ 方法1：使用空字符串表示根目录
            print("尝试 1: path=''")
            result = await list_tool.ainvoke({"path": ""})
            print(f"结果: {result}")

            # ✅ 方法2：使用相对路径（不带 ./）
            print("\n尝试 2: path='.'")
            result = await list_tool.ainvoke({"path": "."})
            print(f"结果: {result}")

            # ✅ 方法3：使用相对于根目录的路径
            print("\n尝试 3: path='test.txt'")
            # 先检查文件是否存在
            read_tool = next((t for t in tools if t.name == "read_text_file"), None)
            if read_tool:
                file_result = await read_tool.ainvoke({"path": "test.txt"})
                print(f"读取 test.txt: {file_result[:50]}...")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(test_mcp())