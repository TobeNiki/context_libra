
import functools
from typing import Generator, List
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation

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



import numpy as np
from typing import Union
# https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/utils/math.py
Matrix = Union[list[List[float]], List[np.ndarray], np.ndarray]
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