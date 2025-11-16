
from typing import List, Dict, Generator, Union, cast, Tuple, Literal
import functools
import numpy as np
from src.embedding_generator import EmbeddingGenerator
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation


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

            # Append cosine distance to the list
            distances.append(distance)

            # Store distance in the dictionary
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

            # Slice the sentence_dicts from the current start index to the end index
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
        


def ja_sentence_splitter_init() -> Generator[str, None, None]:
    # segmenter の定義
    split_punc2 = functools.partial(
        split_punctuation, 
        punctuations=r".。!?"
    )
    concat_tail_no = functools.partial(
        concatenate_matching, 
        former_matching_rule=r"^(?P<result>.+)(の)$", 
        remove_former_matched=False
    )
    concat_tail_te = functools.partial(
        concatenate_matching, 
        former_matching_rule=r"^(?P<result>.+)(て)$", 
        remove_former_matched=False
        )
    concat_decimal = functools.partial(
        concatenate_matching, 
        former_matching_rule=r"^(?P<result>.+)(\d.)$", 
        latter_matching_rule=r"^(\d)(?P<result>.+)$", 
        remove_former_matched=False, 
        remove_latter_matched=False
    )
    return make_pipeline(
        normalize, 
        split_newline, 
        concat_tail_no, 
        concat_tail_te, 
        split_punc2, 
        concat_decimal
    )
# https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/utils/math.py
Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]
def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - np.array(simd.cdist(X, Y, metric="cosine"))
        return Z
    except ImportError:
        
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity