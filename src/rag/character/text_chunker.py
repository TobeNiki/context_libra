import logging

class TextChunker:
    def __init__(self):
        """
        DocumentProcessorのコンストラクタ
        """
        # ロガーの設定
        self.logger = logging.getLogger("document_processor")
        self.logger.setLevel(logging.INFO)

    def split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
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