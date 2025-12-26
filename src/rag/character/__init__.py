
from src.rag.character.character_rag_service import CharacterRagService
from src.rag.character.document_processor import DocumentProcessor
from src.rag.character.embedding_generator import EmbeddingGenerator
from src.rag.character.hybrid_search_database import HybridSearchDatabase

import os

def create_rag_service_from_env() -> CharacterRagService:
    """
    環境変数からRAGサービスを作成します。

    Returns:
        RAGサービスのインスタンス
    """
    # 環境変数から接続情報を取得
    postgres_host = os.environ.get("POSTGRES_HOST", "localhost")
    postgres_port = os.environ.get("POSTGRES_PORT", "5432")
    postgres_user = os.environ.get("POSTGRES_USER", "postgres")
    postgres_password = os.environ.get("POSTGRES_PASSWORD", "password")
    postgres_db = os.environ.get("POSTGRES_DB", "ragdb")

    embedding_model = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

    # コンポーネントの作成
    document_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator(model_name=embedding_model)
    vector_database = HybridSearchDatabase(
        {
            "host": postgres_host,
            "port": postgres_port,
            "user": postgres_user,
            "password": postgres_password,
            "database": postgres_db,
        }
    )

    # RAGサービスの作成
    rag_service = CharacterRagService(document_processor, embedding_generator, vector_database)

    return rag_service