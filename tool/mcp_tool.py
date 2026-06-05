from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import List
from langchain_core.tools import BaseTool


class MCPToolLoader:
    """文件系统 MCP 工具加载器"""

    def __init__(self, root_dir: str = "."):
        """
        初始化文件系统 MCP 客户端

        Args:
            root_dir: 文件系统服务的根目录，默认为当前目录
        """
        self.client = MultiServerMCPClient(
            {
                "filesystem": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        root_dir,  # 可配置的根目录
                    ],
                },
            }
        )

    async def get_tools(self) -> List[BaseTool]:
        """获取文件系统工具列表（read_file、write_file、list_directory 等）"""
        return await self.client.get_tools()

    async def close(self):
        """关闭连接"""
        await self.client.close()


# 便捷函数
async def load_filesystem_tools(root_dir: str = ".") -> List[BaseTool]:
    """
    快速加载文件系统 MCP 工具

    Args:
        root_dir: 文件系统访问的根目录

    Returns:
        List[BaseTool]: 工具列表，包含 read_file、write_file、list_directory 等
    """
    loader = MCPToolLoader(root_dir=root_dir)
    return await loader.get_tools()