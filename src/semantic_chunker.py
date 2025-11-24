
from typing import List, Dict, cast, Tuple, Literal
import numpy as np
from src.embedding_generator import EmbeddingGenerator
from src.utils import ja_sentence_splitter_init, cosine_similarity

BreakpointThresholdType = Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
    "gradient": 95,
}    

class SemanticChunker:
    # https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/text_splitter.py
    def __init__(
        self,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: float | None = None,
        min_chunk_size: int | None = None,
    ):
        self.breakpoint_threshold_type = breakpoint_threshold_type
        if breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[
                breakpoint_threshold_type
            ]
        else:
            self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size
        # 日本語テキスト分割
        self.ja_sentence_splitter = ja_sentence_splitter_init()
        self.embedding_generator = EmbeddingGenerator()
    

    def _calculate_breakpoint_threshold(
        self, distances: List[float],
    ) -> Tuple[float, List[float]]:
        if self.breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, self.breakpoint_threshold_amount),
            ), distances
        elif self.breakpoint_threshold_type == "standard_deviation":
            return cast(
                float,
                np.mean(distances)
                + self.breakpoint_threshold_amount * np.std(distances),
            ), distances
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1

            return np.mean(
                distances
            ) + self.breakpoint_threshold_amount * iqr, distances
        elif self.breakpoint_threshold_type == "gradient":
            # Calculate the threshold based on the distribution of gradient of distance array. # noqa: E501
            distance_gradient = np.gradient(distances, range(0, len(distances)))
            return cast(
                float,
                np.percentile(distance_gradient, self.breakpoint_threshold_amount),
            ), distance_gradient
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )
    
    def split_chunks(self, text: str) -> List[str]:
        """
        テキストをチャンクに分割します。

        Args:
            text: 分割するテキスト
        
        Returns:
            チャンクのリスト
        """

        if not text:
            return []
        
        split_texts = list(self.ja_sentence_splitter(text))

        sentences = [
            {
                'sentence': text,
                'embedding': self.embedding_generator.generate_embedding(text)
            }
            for text in split_texts
        ]
        
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['embedding']
            embedding_next = sentences[i + 1]['embedding']

            # Calculate cosine similarity
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            
            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the List
            distances.append(distance)

            # Store distance in the Dictionary
            sentences[i]["distance_to_next"] = distance

        (
            breakpoint_distance_threshold,
            breakpoint_array,
        ) = self._calculate_breakpoint_threshold(distances)
        indices_above_thresh = [
            i
            for i, x in enumerate(breakpoint_array)
            if x > breakpoint_distance_threshold
        ]

        chunks = []
        start_index = 0

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_Dicts from the current start index to the end index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            # If specified, merge together small chunks.
            if (
                self.min_chunk_size is not None
                and len(combined_text) < self.min_chunk_size
            ):
                continue
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)
        return chunks
        
