from pydantic import BaseModel
from agents import Agent, function_tool
from src.rag.character import create_rag_service_from_env

PROMPT = (
    "あなたは小説のシーンの適切な描写を手伝うアシスタントです。"
    "ユーザがシーンの描画に必要な全ての登場人物の設定について質問した場合、"
    "回答を登場人物名と、その人物ごとの情報で整理した形式で記述し、回答を裏付ける出典を提供してください。"
)

rag = create_rag_service_from_env()

@function_tool
async def character_setting_search(query: str) -> str:
    """
    このツールは小説のシーンの適切な描写に必要な登場人物の設定をまとめたデータへの検索機能を提供します。
    このツールを利用する際は、検索クエリに可能な限り多くのコンテキスト(例えば登場人物名など)を含め、自然言語でクエリを記述してください。
    本ツールはクエリに最も関連性の高い文書断片を5件返します。
    これには、結果の類似度スコア、テキストコンテンツが含まれます。
    """
    doc_count = rag.get_document_count()
    if doc_count == 0:
        return "ドキュメントが存在しませんでした."
    
    results = rag.search(
        query=query,
        limit=5,
        with_context=False,
        full_document=False,
    )
    setting_knowledge = """
    | Score | Content |
    |-------|---------|
    """
    for result in results:
        setting_knowledge += (
            f"| {result['similarity']:.2f} "
            "| " + result['content'].replace('|', '\|') + " "
        )
    return setting_knowledge


class CharacterSetting(BaseModel):
    setting: str
    """登場人物の設定情報"""

    name: str
    """登場人物名"""


scene_writing_assistant_agent = Agent(
    name="Scene Writing Assistant Agent",
    instructions=PROMPT,
    model="gpt-4.1-mini",
    tools=[character_setting_search],
    output_type=CharacterSetting,
)