from agents import FunctionTool,  RunContextWrapper, function_tool, Agent, gen_trace_id, trace
from pydantic import BaseModel
from typing_extensions import Any, TypedDict
from .rag.rag_tools import create_rag_service_from_env
import os 

class SearchParams(TypedDict):
    query: str
    limit: int

rag_service = create_rag_service_from_env()

@function_tool
async def search_rag(params: SearchParams) -> str:
    query = params.get("query")
    limit = params.get("limit")
    if not query:
        return "エラー: 検索クエリが指定されていません"
    
    try:
        # ドキュメント数を確認
        doc_count = rag_service.get_document_count()
        if doc_count == 0:
            return "インデックスにドキュメントが存在しません。CLIコマンド `python -m src.cli index` を使用してドキュメントをインデックス化してください。"
        
        # 検索を実行（前後のチャンクも取得、ドキュメント全体も取得）
        results = rag_service.search(query, limit)

        if not results:
            return f"クエリ '{query}' に一致する結果が見つかりませんでした"
        source_knowledge = """
        | Similarity | Content | File |
        |-------|---------|-------|
        """
        for result in results:
            source_knowledge += (
                f"| {result['similarity']:.2f} "
                "| " + result['content'].replace('|', '\|') + " "
                f"| {result['file_path']} "
        )
        return source_knowledge

    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"検索中にエラーが発生しました: {str(e)}",
                }
            ],
            "isError": True,
        }

agent = Agent(
    name="サンプル",
    model="gpt-4o-mini",
    tools=[search_rag]
)

from rich.console import Console
from .printer import Printer
from agents import LocalShellExecutor
from pathlib import Path
import os

class Ghostwriter:

    def __init__(self):
        self.console = Console()
        self.printer = Printer(self.console)

    async def run(self, query: str):
        trace_id = gen_trace_id()
        with trace("ghostwriter trace", trace_id=trace_id):
            self.printer.update_item(
                "trace_id",
                f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}",
                is_done=True,
                hide_checkmark=True,
            )
            self.printer.update_item("start", "Starting generate writing...", is_done=True)
            

            self.printer.end()

    async def _check_steering(self):
        BASE_DIR = Path(__file__).resolve().parent
        thema_file_path = BASE_DIR / "steering/thema.md"

        if not thema_file_path.is_file():
            raise FileNotFoundError(f"{thema_file_path}が見つかりません")
        
        
        thema_text = thema_file_path.read_text(encoding='utf-8')


    async def _planning(self):
        pass 
    
    async def _get_charactor_info(self):
        pass 

    async def _understand_immediate_context(self):
        pass

    async def _writing_scene(self):
        pass

    async def _feedback_by_llm(self):
        """
        _feedback_by_llm の Docstring
        
        :param self: 説明

        1. 直前の話や設計書的に矛盾していないかをチェック
        2. キャラクターの性格や情報に間違いがないか
        3. キャラクターの視点で考えていること（サリアンチェック）
            - 直前の話での登場人物の行動、心情から、描写したシーンが矛盾していないかをチェック
        """
        pass

    async def _document_proofreading(self):
        pass 

    async def _update_steering(self):
        pass




