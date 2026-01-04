import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Generator, Union, cast, Tuple
import hashlib
import time
import numpy as np
import markitdown
from src.rag.character.semantic_chunker import SemanticChunker
from src.rag.character.text_chunker import TextChunker

class DocumentProcessor:
    """
    ドキュメント処理クラス

    マークダウン、テキスト、パワーポイント、PDFなどのファイルの読み込みと解析、チャンク分割を行います。

    Attributes:
        logger: ロガー
    """

    # サポートするファイル拡張子
    SUPPORTED_EXTENSIONS = {
        "text": [".txt", ".md", ".markdown"],
        "office": [".ppt", ".pptx", ".doc", ".docx"],
        "pdf": [".pdf"],
    }

    def __init__(self, text_chunker: TextChunker):
        """コンストラクタ"""
        # ロガーの設定
        self.logger = logging.getLogger("document_processor")
        self.logger.setLevel(logging.INFO)
        self.text_chunker = text_chunker

    def read_file(self, file_path: str) -> str:
        """
        ファイルを読み込みます。

        Args:
            file_path: ファイルのパス

        Returns:
            ファイルの内容

        Raises:
            FileNotFoundError: ファイルが見つからない場合
            IOError: ファイルの読み込みに失敗した場合
        """
        try:
            ext = Path(file_path).suffix.lower()

            # テキストファイル（マークダウン含む）の場合
            if ext in self.SUPPORTED_EXTENSIONS["text"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # NUL文字を削除
                    content = content.replace("\x00", "")
                self.logger.info(f"テキストファイル '{file_path}' を読み込みました")
                return content

            # パワーポイント、Word、PDFの場合はmarkitdownを使用して変換
            elif ext in self.SUPPORTED_EXTENSIONS["office"] or ext in self.SUPPORTED_EXTENSIONS["pdf"]:
                return self._convert_to_markdown(file_path)

            # サポートしていない拡張子の場合
            else:
                self.logger.warning(f"サポートしていないファイル形式です: {file_path}")
                return ""

        except FileNotFoundError:
            self.logger.error(f"ファイル '{file_path}' が見つかりません")
            raise
        except IOError as e:
            self.logger.error(f"ファイル '{file_path}' の読み込みに失敗しました: {str(e)}")
            raise

    def _convert_to_markdown(self, file_path: str) -> str:
        """
        パワーポイント、Word、PDFなどのファイルをマークダウンに変換します。

        Args:
            file_path: ファイルのパス

        Returns:
            マークダウンに変換された内容

        Raises:
            Exception: 変換に失敗した場合
        """
        try:
            # ファイルURIを作成
            file_uri = f"file://{os.path.abspath(file_path)}"

            # markitdownを使用して変換
            markdown_content = markitdown.MarkItDown().convert_uri(file_uri).markdown
            # NUL文字を削除
            markdown_content = markdown_content.replace("\x00", "")

            self.logger.info(f"ファイル '{file_path}' をマークダウンに変換しました")
            return markdown_content
        except Exception as e:
            self.logger.error(f"ファイル '{file_path}' のマークダウン変換に失敗しました: {str(e)}")
            raise

    def calculate_file_hash(self, file_path: str) -> str:
        """
        ファイルのハッシュ値を計算します。

        Args:
            file_path: ファイルのパス

        Returns:
            ファイルのSHA-256ハッシュ値
        """
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            self.logger.error(f"ファイル '{file_path}' のハッシュ計算に失敗しました: {str(e)}")
            # エラーが発生した場合は、タイムスタンプをハッシュとして使用
            return f"timestamp-{int(time.time())}"

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        ファイルのメタデータを取得します。

        Args:
            file_path: ファイルのパス

        Returns:
            ファイルのメタデータ（ハッシュ値、最終更新日時など）
        """
        file_stat = os.stat(file_path)
        return {
            "hash": self.calculate_file_hash(file_path),
            "mtime": file_stat.st_mtime,
            "size": file_stat.st_size,
            "path": file_path,
        }

    def load_file_registry(self, processed_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        処理済みファイルのレジストリを読み込みます。

        Args:
            processed_dir: 処理済みファイルを保存するディレクトリのパス

        Returns:
            処理済みファイルのレジストリ（ファイルパスをキーとするメタデータの辞書）
        """
        registry_path = Path(processed_dir) / "file_registry.json"
        if not registry_path.exists():
            return {}

        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"ファイルレジストリの読み込みに失敗しました: {str(e)}")
            return {}

    def save_file_registry(self, processed_dir: str, registry: Dict[str, Dict[str, Any]]) -> None:
        """
        処理済みファイルのレジストリを保存します。

        Args:
            processed_dir: 処理済みファイルを保存するディレクトリのパス
            registry: 処理済みファイルのレジストリ
        """
        registry_path = Path(processed_dir) / "file_registry.json"
        try:
            # 処理済みディレクトリが存在しない場合は作成
            os.makedirs(Path(processed_dir), exist_ok=True)

            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)
            self.logger.info(f"ファイルレジストリを保存しました: {registry_path}")
        except Exception as e:
            self.logger.error(f"ファイルレジストリの保存に失敗しました: {str(e)}")

    def process_file(
        self, file_path: str, processed_dir: str, chunk_size: int = 500, overlap:int = 100,
    ) -> List[Dict[str, Any]]:
        """
        ファイルを処理します。

        Args:
            file_path: ファイルのパス
            processed_dir: 処理済みファイルを保存するディレクトリのパス

        Returns:
            処理結果のリスト（各要素はチャンク情報を含む辞書）
        """
        try:
            # ファイルを読み込む
            content = self.read_file(file_path)
            if not content:
                return []

            # ファイルパスからディレクトリ構造を取得
            file_path_obj = Path(file_path)
            relative_path = file_path_obj.relative_to(Path(file_path_obj.parts[0]) / Path(file_path_obj.parts[1]))
            parent_dirs = relative_path.parent.parts

            # ディレクトリ名をサフィックスとして使用
            dir_suffix = "_".join(parent_dirs) if parent_dirs else ""

            # 処理済みファイル名を生成
            processed_file_name = f"{file_path_obj.stem}{('_' + dir_suffix) if dir_suffix else ''}.md"
            processed_file_path = Path(processed_dir) / processed_file_name

            # 処理済みディレクトリが存在しない場合は作成
            os.makedirs(Path(processed_dir), exist_ok=True)

            # 処理済みファイルに書き込む
            with open(processed_file_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.logger.info(f"処理済みファイルを保存しました: {processed_file_path}")

            # チャンクに分割
            chunks = self.text_chunker.split_into_chunks(content, chunk_size=chunk_size, overlap=overlap)

            # 結果を作成
            results = []
            for i, chunk in enumerate(chunks):
                document_id = f"{processed_file_name}_{i}"
                results.append(
                    {
                        "document_id": document_id,
                        "content": chunk,
                        "file_path": str(processed_file_path),
                        "original_file_path": file_path,
                        "chunk_index": i,
                        "metadata": {
                            "file_name": file_path_obj.name,
                            "directory": str(file_path_obj.parent),
                            "directory_suffix": dir_suffix,
                        },
                    }
                )

            self.logger.info(f"ファイル '{file_path}' を処理しました（{len(results)} チャンク）")
            return results

        except Exception as e:
            self.logger.error(f"ファイル '{file_path}' の処理中にエラーが発生しました: {str(e)}")
            raise

    def process_directory(
        self, source_dir: str, processed_dir: str, incremental: bool = False
    ) -> List[Dict[str, Any]]:
        """
        ディレクトリ内のファイルを処理します。

        Args:
            source_dir: 原稿ファイルが含まれるディレクトリのパス
            processed_dir: 処理済みファイルを保存するディレクトリのパス
            incremental: 差分のみを処理するかどうか

        Returns:
            処理結果のリスト（各要素はチャンク情報を含む辞書）
        """
        results = []
        source_directory = Path(source_dir)

        if not source_directory.exists() or not source_directory.is_dir():
            self.logger.error(f"ディレクトリ '{source_dir}' が見つからないか、ディレクトリではありません")
            raise FileNotFoundError(f"ディレクトリ '{source_dir}' が見つからないか、ディレクトリではありません")

        # サポートするファイル拡張子を全て取得
        all_extensions = []
        for ext_list in self.SUPPORTED_EXTENSIONS.values():
            all_extensions.extend(ext_list)

        # ファイルを検索
        files = []
        for ext in all_extensions:
            files.extend(list(source_directory.glob(f"**/*{ext}")))

        self.logger.info(f"ディレクトリ '{source_dir}' 内に {len(files)} 個のファイルが見つかりました")

        # 差分処理の場合、ファイルレジストリを読み込む
        if incremental:
            file_registry = self.load_file_registry(processed_dir)
            self.logger.info(f"ファイルレジストリから {len(file_registry)} 個のファイル情報を読み込みました")
        else:
            file_registry = {}

        # 処理対象のファイルを特定
        files_to_process = []
        for file_path in files:
            str_path = str(file_path)
            if incremental:
                # ファイルのメタデータを取得
                current_metadata = self.get_file_metadata(str_path)

                # レジストリに存在しない、またはハッシュ値が変更されている場合のみ処理
                if (
                    str_path not in file_registry
                    or file_registry[str_path]["hash"] != current_metadata["hash"]
                    or file_registry[str_path]["mtime"] != current_metadata["mtime"]
                    or file_registry[str_path]["size"] != current_metadata["size"]
                ):
                    files_to_process.append(file_path)
                    # レジストリを更新
                    file_registry[str_path] = current_metadata
            else:
                # 差分処理でない場合は全てのファイルを処理
                files_to_process.append(file_path)
                # レジストリを更新
                file_registry[str_path] = self.get_file_metadata(str_path)

        self.logger.info(f"処理対象のファイル数: {len(files_to_process)} / {len(files)}")

        # 各ファイルを処理
        for file_path in files_to_process:
            try:
                file_results = self.process_file(str(file_path), processed_dir)
                results.extend(file_results)
            except Exception as e:
                self.logger.error(f"ファイル '{file_path}' の処理中にエラーが発生しました: {str(e)}")
                # エラーが発生しても処理を続行
                continue

        # ファイルレジストリを保存
        self.save_file_registry(processed_dir, file_registry)

        self.logger.info(f"ディレクトリ '{source_dir}' 内のファイルを処理しました（合計 {len(results)} チャンク）")
        return results