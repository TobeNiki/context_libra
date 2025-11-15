import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from sudachipy import tokenizer, dictionary
import hashlib
import time

import markitdown

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

    def __init__(self):
        """コンストラクタ"""
        # ロガーの設定
        self.logger = logging.getLogger("document_processor")
        self.logger.setLevel(logging.INFO)

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
    
    def _split_into_chunks(self, text:str, chunk_size: int = 500, overlap: int = 100, splitmode: str = "C") -> List[str]:
        """
        テキストをチャンクに分割します。

        Args:
            text: 分割するテキスト
            chunk_size: チャンクサイズ（文字数）
            overlap: チャンク間のオーバーラップ（文字数）

        Returns:
            チャンクのリスト
        """
        if not text:
            return []
        
        token_obj = dictionary.Dictionary().create()
        mode_map = {
            "A": tokenizer.Tokenizer.SplitMode.A,
            "B": tokenizer.Tokenizer.SplitMode.B,
            "C": tokenizer.Tokenizer.SplitMode.C,
        }
        mode_key = splitmode.upper()
        mode = mode_map.get(mode_key, tokenizer.Tokenizer.SplitMode.C)
        self.logger.info(f"SudachiPy SplitMode: {mode_key} ({mode})")

        sentences = []
        buf = ""
        for line in text.splitlines():
            for c in line:
                buf += c
                if c in "。！？\n":
                    sentences.append(buf.strip())
                    buf = ""
            if buf:
                sentences.append(buf.strip())
                buf = ""
        # 文節ごとに分割
        chunks = []
        for sentence in sentences:
            morphemes = token_obj.tokenize(sentence, mode)
            chunk = "".join([m.surface() for m in morphemes])
            if chunk:
                chunks.append(chunk)
        return chunks

    def split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        テキストをチャンクに分割します。

        Args:
            text: 分割するテキスト
            chunk_size: チャンクサイズ（文字数）
            overlap: チャンク間のオーバーラップ（文字数）

        Returns:
            チャンクのリスト
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            # 文の途中で切らないように調整
            if end < text_length:
                # 次の改行または句点を探す
                next_newline = text.find("\n", end)
                next_period = text.find("。", end)

                if next_newline != -1 and (next_period == -1 or next_newline < next_period):
                    end = next_newline + 1  # 改行を含める
                elif next_period != -1:
                    end = next_period + 1  # 句点を含める

            chunks.append(text[start:end])
            start = end - overlap if end - overlap > start else end

            # 終了条件
            if start >= text_length:
                break

        self.logger.info(f"テキストを {len(chunks)} チャンクに分割しました")
        return chunks
