import logging
import psycopg2
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Literal
from ranx import Run, fuse, optimize_fusion, Qrels

load_dotenv()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

class HybridSearchDatabase:
    """
    ベクトル+全文検索のハイブリッド検索のデータベースクラス

    PostgreSQLとpgvectorを使用してベクトルの保存と検索を行います。

    Attributes:
        connection_params: 接続パラメータ
        connection: データベース接続
        logger: ロガー
    """

    def __init__(self, connection_params: dict[str, Any]):
        """
        HybridSearchDatabaseのコンストラクタ

        Args:
            connection_params: 接続パラメータ
                - host: ホスト名
                - port: ポート番号
                - user: ユーザー名
                - password: パスワード
                - database: データベース名
        """
        # ロガーの設定
        self.logger = logging.getLogger("vector_database")
        self.logger.setLevel(logging.INFO)

        # 接続パラメータの保存
        self.connection_params = connection_params
        self.connection = None

    def connect(self):
        """
        データベースに接続します。

        Raises:
            Exception: 接続に失敗した場合
        """
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            self.logger.info("データベースに接続しました")
        except Exception as e:
            self.logger.error(f"データベースへの接続に失敗しました: {str(e)}")
            raise

    def disconnect(self):
        """
        データベースから切断する
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("データベースから切断しました")

    def initialize_database(self) -> None:
        """
        データベースを初期化します。

        テーブルとインデックスを作成します。

        Raises:
            Exception: 初期化に失敗した場合
        """
        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # カーソルの作成
            cursor = self.connection.cursor()
            # pgvectorエクステンションの有効化
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # pgoongaエクステンションの有効化
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pgroonga;")

            # ドキュメントテーブルの作成
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    document_id TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    metadata JSONB,
                    embedding vector({EMBEDDING_DIM}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # インデックスの作成
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_document_id ON documents (document_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents (file_path);
            """)
            # ベクトル検索
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents
                    USING ivfflat (embedding vector_cosine_ops);
            """)
            # PGroongaインデックスを作成（日本語トークナイザーを使用）
            # TokenMecabは日本語の形態素解析を行い、名詞、動詞、形容詞などを正確に認識します
            # normalizer='NormalizerNFKC150'で正規化も行います
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_documents_pgroonga ON documents
                    USING pgroonga (content)
                    WITH (tokenizer='TokenMecab',
                        normalizer='NormalizerNFKC150');
            """)

            # コミット
            self.connection.commit()
            self.logger.info("データベースを初期化しました")

        except Exception as e:
            # ロールバック
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"データベースの初期化に失敗しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def insert_document(
        self,
        document_id: str,
        content: str,
        file_path: str,
        chunk_index: int,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        ドキュメントを挿入します。

        Args:
            document_id: ドキュメントID
            content: ドキュメントの内容
            file_path: ファイルパス
            chunk_index: チャンクインデックス
            embedding: エンベディング
            metadata: メタデータ（オプション）

        Raises:
            Exception: 挿入に失敗した場合
        """
        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # カーソルの作成
            cursor = self.connection.cursor()

            # メタデータをJSON形式に変換
            metadata_json = json.dumps(metadata) if metadata else None

            # ドキュメントの挿入
            cursor.execute(
                """
                INSERT INTO documents (document_id, content, file_path, chunk_index, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    file_path = EXCLUDED.file_path,
                    chunk_index = EXCLUDED.chunk_index,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    created_at = CURRENT_TIMESTAMP;
            """,
                (document_id, content, file_path, chunk_index, embedding, metadata_json),
            )

            # コミット
            self.connection.commit()
            self.logger.debug(f"ドキュメント '{document_id}' を挿入しました")

        except Exception as e:
            # ロールバック
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"ドキュメントの挿入に失敗しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def batch_insert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        複数のドキュメントをバッチ挿入します。

        Args:
            documents: ドキュメントのリスト
                各ドキュメントは以下のキーを持つ辞書:
                - document_id: ドキュメントID
                - content: ドキュメントの内容
                - file_path: ファイルパス
                - chunk_index: チャンクインデックス
                - embedding: エンベディング
                - metadata: メタデータ（オプション）

        Raises:
            Exception: 挿入に失敗した場合
        """
        if not documents:
            self.logger.warning("挿入するドキュメントがありません")
            return

        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # カーソルの作成
            cursor = self.connection.cursor()

            # バッチ挿入用のデータ作成
            values = []
            for doc in documents:
                metadata_json = json.dumps(doc.get("metadata")) if doc.get("metadata") else None
                values.append(
                    (doc["document_id"], doc["content"], doc["file_path"], doc["chunk_index"], doc["embedding"], metadata_json)
                )

            # セッション単位でメモリ増加
            cursor.execute("SET maintenance_work_mem = '1GB';")

            # バッチ挿入
            cursor.executemany(
                """
                INSERT INTO documents (document_id, content, file_path, chunk_index, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    file_path = EXCLUDED.file_path,
                    chunk_index = EXCLUDED.chunk_index,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    created_at = CURRENT_TIMESTAMP;
            """,
                values,
            )

            # コミット
            self.connection.commit()
            self.logger.info(f"{len(documents)} 個のドキュメントを挿入しました")

        except Exception as e:
            # ロールバック
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"ドキュメントのバッチ挿入に失敗しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def search_by_vector(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        ベクトル検索を行います。

        Args:
            query_embedding: クエリのエンベディング
            limit: 返す結果の数（デフォルト: 5）

        Returns:
            検索結果のリスト（関連度順）

        Raises:
            Exception: 検索に失敗した場合
        """
        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # カーソルの作成
            cursor = self.connection.cursor()

            # クエリエンベディングをPostgreSQLの配列構文に変換
            embedding_str = str(query_embedding)
            embedding_array = f"ARRAY{embedding_str}::vector"
            # ベクトル検索
            cursor.execute(
                f"""
                SELECT
                    document_id,
                    content,
                    file_path,
                    chunk_index,
                    metadata,
                    1 - (embedding <=> {embedding_array}) AS similarity
                FROM
                    documents
                WHERE
                    embedding IS NOT NULL
                ORDER BY
                    similarity desc
                LIMIT %s;
                """,
                (limit,),
            )
            # 結果の取得
            results: List[Dict[str, Any]] = []
            for i, row in enumerate(cursor.fetchall()):
                document_id, content, file_path, chunk_index, metadata_json, similarity = row

                # メタデータをJSONからデコード
                if metadata_json:
                    if isinstance(metadata_json, str):
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        # 既に辞書型の場合はそのまま使用
                        metadata = metadata_json
                else:
                    metadata = {}

                results.append(
                    {
                        "document_id": document_id,
                        "content": content,
                        "file_path": file_path,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                        "similarity": similarity,
                        "rank": i + 1,
                    }
                )
            self.logger.info(f"クエリに対して {len(results)} 件の結果が見つかりました")
            return results

        except Exception as e:
            self.logger.error(f"ベクトル検索中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def tokenize_query(self, query: str, expr: Literal["OR", "AND"] = "OR") -> str:
        """
        形態素解析(tokenize)して全文検索用のクエリを形成

        Args:
            query: クエリ
            expr: OR AND

        Returns:
            全文検索用のクエリ

        Raises:
            Exception: 検索に失敗した場合
        """
        if expr not in ["OR", "AND"]:
            raise ValueError()

        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # カーソルの作成
            cursor = self.connection.cursor()

            # pgroonga_tokenizeを使って形態素解析(tokenizerはmecab)
            cursor.execute("""
                select pgroonga_tokenize(%s, 'tokenizer', 'TokenMecab', 'normalizer', 'NormalizerNFKC150')
                """,
                (query),
            )
            # 結果
            result = cursor.fetchone()
            if not result:
                return []

            raw_tokens = result[0]  # PostgreSQL text[] → Python list[str]

            tokens: List[Dict[str, Any]] = []
            for token_str in raw_tokens:
                try:
                    # text[] の各要素は JSON string なので json.loads する
                    token_obj = json.loads(token_str)
                    tokens.append(token_obj)
                except Exception as e:
                    # JSON パース失敗時はスキップ
                    self.logger.warning(f"トークンの JSON パースに失敗: {token_str} ({e})")

            self.logger.info(f"トークン化 '{query}' → {tokens}")

            expr_str = expr if expr == "OR" else " "
            return expr_str.join(map(lambda token : token['value'], tokens))

        except Exception as e:
            self.logger.error(f"解析中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()


    def search_by_fulltext(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        PGroonga を利用して全文検索を行います。

        Args:
            query: 検索クエリ文字列（PGroonga の &@~ 演算子に渡す文字列）
            limit: 返す結果の数（デフォルト: 5）

        Returns:
            検索結果のリスト（search_by_vector と同じ構造）
                - document_id
                - content
                - file_path
                - chunk_index
                - metadata (dict)
                - similarity (PGroonga のスコア)
        """
        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # カーソルの作成
            cursor = self.connection.cursor()

            # PGroonga による全文検索
            # content &@~ %s で複数キーワード（OR/AND）を含む検索が可能
            # pgroonga_score(tableoid, ctid) でマッチ度スコアを取得
            cursor.execute(
                """
                SELECT
                    document_id,
                    content,
                    file_path,
                    chunk_index,
                    metadata,
                    pgroonga_score(tableoid, ctid) AS similarity
                FROM
                    documents
                WHERE
                    content &@~ %s
                ORDER BY
                    similarity DESC
                LIMIT %s;
                """,
                (query, limit),
            )

            # 結果の取得（search_by_vector と同じ形式に整形）
            results: List[Dict[str, Any]] = []
            for i, row in enumerate(cursor.fetchall()):
                document_id, content, file_path, chunk_index, metadata_json, similarity = row

                # メタデータをJSONからデコード（既存メソッドと同じロジック）
                if metadata_json:
                    if isinstance(metadata_json, str):
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        # 既に辞書型の場合はそのまま使用
                        metadata = metadata_json
                else:
                    metadata = {}

                results.append(
                    {
                        "document_id": document_id,
                        "content": content,
                        "file_path": file_path,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                        "similarity": similarity,
                        "rank": i + 1,
                    }
                )

            self.logger.info(f"全文検索クエリ '{query}' に対して {len(results)} 件の結果が見つかりました")
            return results

        except Exception as e:
            self.logger.error(f"全文検索中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()


    def delete_document(self, document_id: str) -> bool:
        """
        ドキュメントを削除します。

        Args:
            document_id: 削除するドキュメントのID

        Returns:
            削除に成功した場合はTrue、ドキュメントが見つからない場合はFalse

        Raises:
            Exception: 削除に失敗した場合
        """
        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # カーソルの作成
            cursor = self.connection.cursor()

            # ドキュメントの削除
            cursor.execute("DELETE FROM documents WHERE document_id = %s;", (document_id,))

            # 削除された行数を取得
            deleted_rows = cursor.rowcount

            # コミット
            self.connection.commit()

            if deleted_rows > 0:
                self.logger.info(f"ドキュメント '{document_id}' を削除しました")
                return True
            else:
                self.logger.warning(f"ドキュメント '{document_id}' が見つかりません")
                return False

        except Exception as e:
            # ロールバック
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"ドキュメントの削除中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def delete_by_file_path(self, file_path: str) -> int:
        """
        ファイルパスに基づいてドキュメントを削除します。

        Args:
            file_path: 削除するドキュメントのファイルパス

        Returns:
            削除されたドキュメントの数

        Raises:
            Exception: 削除に失敗した場合
        """
        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # カーソルの作成
            cursor = self.connection.cursor()

            # ドキュメントの削除
            cursor.execute("DELETE FROM documents WHERE file_path = %s;", (file_path,))

            # 削除された行数を取得
            deleted_rows = cursor.rowcount

            # コミット
            self.connection.commit()

            self.logger.info(f"ファイルパス '{file_path}' に関連する {deleted_rows} 個のドキュメントを削除しました")
            return deleted_rows

        except Exception as e:
            # ロールバック
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"ドキュメントの削除中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def clear_database(self) -> int:
        """
        データベースをクリアします（全てのドキュメントを削除）。

        Raises:
            Exception: クリアに失敗した場合

        Returns:
            削除されたドキュメントの数。テーブルをDROPするため、削除前の数を返します。
        """
        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # 削除前のドキュメント数を取得
            count_before_delete = self.get_document_count()

            # カーソルの作成
            cursor = self.connection.cursor()

            # テーブルを削除してスキーマもクリア
            cursor.execute("DROP TABLE IF EXISTS documents;")

            # コミット
            self.connection.commit()

            if count_before_delete > 0:
                self.logger.info(
                    f"データベースをクリアしました（documentsテーブルを削除、{count_before_delete} 個のドキュメントが対象でした）"
                )
            else:
                self.logger.info("データベースをクリアしました（documentsテーブルを削除）")
            return count_before_delete

        except Exception as e:
            # ロールバック
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"データベースのクリア中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def get_document_count(self) -> int:
        """
        データベース内のドキュメント数を取得します。

        Returns:
            ドキュメント数

        Raises:
            Exception: 取得に失敗した場合
        """
        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # カーソルの作成
            cursor = self.connection.cursor()

            # ドキュメント数を取得
            cursor.execute("SELECT COUNT(*) FROM documents;")
            count = cursor.fetchone()[0]

            self.logger.info(f"データベース内のドキュメント数: {count}")
            return count

        except psycopg2.errors.UndefinedTable:
            # テーブルが存在しない場合は0を返す
            self.connection.rollback()  # エラー状態をリセット
            self.logger.info("documentsテーブルが存在しないため、ドキュメント数は0です")
            return 0
        except Exception as e:
            self.logger.error(f"ドキュメント数の取得中にエラーが発生しました: {str(e)}")
            raise

    def get_adjacent_chunks(self, file_path: str, chunk_index: int, context_size: int = 1) -> List[Dict[str, Any]]:
        """
        指定されたチャンクの前後のチャンクを取得します。

        Args:
            file_path: ファイルパス
            chunk_index: チャンクインデックス
            context_size: 前後に取得するチャンク数（デフォルト: 1）

        Returns:
            前後のチャンクのリスト

        Raises:
            Exception: 取得に失敗した場合
        """
        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # カーソルの作成
            cursor = self.connection.cursor()

            # 前後のチャンクを取得
            min_index = max(0, chunk_index - context_size)
            max_index = chunk_index + context_size

            cursor.execute(
                """
                SELECT
                    document_id,
                    content,
                    file_path,
                    chunk_index,
                    metadata,
                    1 AS similarity
                FROM
                    documents
                WHERE
                    file_path = %s
                    AND chunk_index >= %s
                    AND chunk_index <= %s
                    AND chunk_index != %s
                ORDER BY
                    chunk_index
                """,
                (file_path, min_index, max_index, chunk_index),
            )

            # 結果の取得
            results = []
            for row in cursor.fetchall():
                document_id, content, file_path, chunk_index, metadata_json, similarity = row

                # メタデータをJSONからデコード
                if metadata_json:
                    if isinstance(metadata_json, str):
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        # 既に辞書型の場合はそのまま使用
                        metadata = metadata_json
                else:
                    metadata = {}

                results.append(
                    {
                        "document_id": document_id,
                        "content": content,
                        "file_path": file_path,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                        "similarity": similarity,
                        "is_context": True,  # コンテキストチャンクであることを示すフラグ
                    }
                )

            self.logger.info(
                f"ファイル '{file_path}' のチャンク {chunk_index} の前後 {len(results)} 件のチャンクを取得しました"
            )
            return results

        except Exception as e:
            self.logger.error(f"前後のチャンク取得中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def get_document_by_file_path(self, file_path: str) -> List[Dict[str, Any]]:
        """
        指定されたファイルパスに基づいてドキュメント全体を取得します。

        Args:
            file_path: ファイルパス

        Returns:
            ドキュメント全体のチャンクのリスト

        Raises:
            Exception: 取得に失敗した場合
        """
        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # カーソルの作成
            cursor = self.connection.cursor()

            # ファイルパスに基づいてドキュメントを取得
            cursor.execute(
                """
                SELECT
                    document_id,
                    content,
                    file_path,
                    chunk_index,
                    metadata,
                    1 AS similarity
                FROM
                    documents
                WHERE
                    file_path = %s
                ORDER BY
                    chunk_index
                """,
                (file_path,),
            )

            # 結果の取得
            results = []
            for row in cursor.fetchall():
                document_id, content, file_path, chunk_index, metadata_json, similarity = row

                # メタデータをJSONからデコード
                if metadata_json:
                    if isinstance(metadata_json, str):
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        # 既に辞書型の場合はそのまま使用
                        metadata = metadata_json
                else:
                    metadata = {}

                results.append(
                    {
                        "document_id": document_id,
                        "content": content,
                        "file_path": file_path,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                        "similarity": similarity,
                        "is_full_document": True,  # 全文ドキュメントであることを示すフラグ
                    }
                )

            self.logger.info(f"ファイル '{file_path}' の全文 {len(results)} チャンクを取得しました")
            return results

        except Exception as e:
            self.logger.error(f"ドキュメント全文の取得中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def maintenance_fulltext_search_index(self):
        try:
            # 接続がない場合は接続
            if not self.connection:
                self.connect()

            # カーソルの作成
            cursor = self.connection.cursor()

            # ファイルパスに基づいてドキュメントを取得
            cursor.execute(
                """
                VACUUM ANALYZE idx_documents_pgroonga;
                """
            )
        except Exception as e:
            self.logger.error(f"全文検索用のインデックス: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def hybrid_search_rrf(
        self,
        query: str,
        query_embedding: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        search_by_fulltext と search_by_vector の結果を
        RRF (Reciprocal Rank Fusion) で統合したハイブリッド検索を行う。

        Args:
            query: 全文検索用のクエリ文字列
            query_embedding: ベクトル検索用のエンベディング
            limit: 返却する件数（両検索ともこの件数で取得）

        Returns:
            search_by_vector / search_by_fulltext と同じ形の辞書リスト。
            similarity には RRF 後のスコアが入る。
        """
        # 1. 個別検索
        fulltext_results = self.search_by_fulltext(query, limit=limit)
        vector_results = self.search_by_vector(query_embedding, limit=limit)

        print(len(fulltext_results))
        print(len(vector_results))
        # どちらも結果がなければ空
        if not fulltext_results and not vector_results:
            return []

        query_id = "q1"

        # 2. ranx 用の Run を作成
        fulltext_run = Run.from_dict(
            {
                query_id: {
                    r["document_id"]: r.get("rank", limit)
                    for r in fulltext_results
                }
            },
            name="fulltext",
        )
        vector_run = Run.from_dict(
            {
                query_id: {
                    r["document_id"]: r.get("rank", limit)
                    for r in vector_results
                }
            },
            name="vector",
        )

        # 3. RRF で融合
        fused_run = fuse(
            runs=[fulltext_run, vector_run],
            method="rrf",   # Reciprocal Rank Fusion
            params={'k': 60}
        )

        fused_scores = fused_run[query_id]  # dict: {document_id: fused_score}

        # 4. document_id -> 元結果 を引けるようにしておく
        doc_index: Dict[str, Dict[str, Any]] = {}
        for r in fulltext_results:
            doc_index.setdefault(r["document_id"], r)
        for r in vector_results:
            doc_index.setdefault(r["document_id"], r)

        # 5. RRF スコアでソートして上位 limit 件を返す
        sorted_items = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

        results: List[Dict[str, Any]] = []
        for doc_id, score in sorted_items:
            base = doc_index.get(doc_id)
            if base is None:
                continue
            item = dict(base)
            item["similarity"] = score  # RRF スコアに差し替え
            results.append(item)

        return results
