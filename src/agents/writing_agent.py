from agents import Agent
PROMPT = (
    "あなたは小説のシーンの適切な描写を行う作家です。"
    "物語のテーマ、登場人物、物、世界観の設定を把握した上で小説のワンシーンを出力します。"
    ""
)

writing_agent = Agent(
    name="Scene Writing Agent",
    instructions=PROMPT,
    model="gpt-4.1-mini",
)

