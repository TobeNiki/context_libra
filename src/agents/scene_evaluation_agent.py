import asyncio
import shutil

from agents import Agent, Runner, trace
from agents.mcp import MCPServer, MCPServerStdio
from agents.items import ToolCallItem, ToolCallOutputItem
PROMPT = (
    "You are an assistant who proofreads and evaluates the text of a scene in a novel. "
    "Textlint is a tool for machine proofreading. "
    "Run Textlint to make sure your text is ok."
)

async def run(mcp_server: MCPServer, output_scene_file_path:str)->str:
    agent = Agent(
        name="Assistant",
        instructions=PROMPT,
        model="gpt-4o-mini",
        mcp_servers=[mcp_server],
    )
    message = f"textlintで {output_scene_file_path} のファイルを校正してください。"

    result = await Runner.run(starting_agent=agent, input=message)
    
    print("=== final_output ===")
    print(result.final_output)

    print("=== new_items (tool related) ===")
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            # ここが「Agentがツールを呼んだ瞬間」
            raw = item.raw_item  # LLMが生成したtool callの生データ
            print("[ToolCallItem]", raw)
        elif isinstance(item, ToolCallOutputItem):
         # ここが「ツールが実行されて結果が返った瞬間」
            raw = item.raw_item  # tool response の生データ
            print("[ToolCallOutputItem]", raw)
    return result.final_output


async def scene_evaluation(output_scene_file_path:str) -> str:

    async with MCPServerStdio(
        cache_tools_list=True,  # Cache the tools list, for demonstration
        params={"command": "npx", "args": ["textlint", "--mcp", "--rule", "textlint-rule-no-dropping-the-ra"]},
    ) as server:
        with trace(workflow_name="MCP Textlint"):
            result = await run(server, output_scene_file_path)
            return result
