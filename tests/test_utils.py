import sys
import types
import pathlib
import numpy as np
import pytest

# utils をインポートする前に外部依存をスタブ化しておく
if "ja_sentence_segmenter" not in sys.modules:
    # パッケージ／サブモジュールを作成して必要な名前を注入
    sys.modules["ja_sentence_segmenter"] = types.ModuleType("ja_sentence_segmenter")

    # common.pipeline.make_pipeline
    pipeline_mod = types.ModuleType("ja_sentence_segmenter.common.pipeline")
    def make_pipeline(*funcs):
        def pipeline(text):
            out = text
            for f in funcs:
                try:
                    out = f(out)
                except Exception:
                    pass
            return out
        return pipeline
    pipeline_mod.make_pipeline = make_pipeline
    sys.modules["ja_sentence_segmenter.common.pipeline"] = pipeline_mod

    # concatenate.simple_concatenator.concatenate_matching
    concat_mod = types.ModuleType("ja_sentence_segmenter.concatenate.simple_concatenator")
    def concatenate_matching(*args, **kwargs):
        def fn(x):
            return x
        return fn
    concat_mod.concatenate_matching = concatenate_matching
    sys.modules["ja_sentence_segmenter.concatenate.simple_concatenator"] = concat_mod

    # normalize.neologd_normalizer.normalize
    norm_mod = types.ModuleType("ja_sentence_segmenter.normalize.neologd_normalizer")
    norm_mod.normalize = lambda x: x
    sys.modules["ja_sentence_segmenter.normalize.neologd_normalizer"] = norm_mod

    # split.simple_splitter.split_newline, split_punctuation
    split_mod = types.ModuleType("ja_sentence_segmenter.split.simple_splitter")
    split_mod.split_newline = lambda x: [x]
    split_mod.split_punctuation = lambda x, punctuations=None: [x]
    sys.modules["ja_sentence_segmenter.split.simple_splitter"] = split_mod

# janome が無い環境向けの最小スタブ
if "janome" not in sys.modules:
    sys.modules["janome"] = types.ModuleType("janome")

    tokenizer_mod = types.ModuleType("janome.tokenizer")
    class Tokenizer:
        def __init__(self, *a, **k): pass
    tokenizer_mod.Tokenizer = Tokenizer
    sys.modules["janome.tokenizer"] = tokenizer_mod

    analyzer_mod = types.ModuleType("janome.analyzer")
    class Analyzer:
        def __init__(self, *a, **k): pass
        def analyze(self, q):
            return []
    analyzer_mod.Analyzer = Analyzer
    sys.modules["janome.analyzer"] = analyzer_mod

    tokenfilter_mod = types.ModuleType("janome.tokenfilter")
    class POSKeepFilter:
        def __init__(self, *a, **k): pass
    class POSStopFilter:
        def __init__(self, *a, **k): pass
    class LowerCaseFilter:
        def __init__(self, *a, **k): pass
    tokenfilter_mod.POSKeepFilter = POSKeepFilter
    tokenfilter_mod.POSStopFilter = POSStopFilter
    tokenfilter_mod.LowerCaseFilter = LowerCaseFilter
    sys.modules["janome.tokenfilter"] = tokenfilter_mod

    charfilter_mod = types.ModuleType("janome.charfilter")
    class UnicodeNormalizeCharFilter:
        def __init__(self, *a, **k): pass
    class RegexReplaceCharFilter:
        def __init__(self, *a, **k): pass
    charfilter_mod.UnicodeNormalizeCharFilter = UnicodeNormalizeCharFilter
    charfilter_mod.RegexReplaceCharFilter = RegexReplaceCharFilter
    sys.modules["janome.charfilter"] = charfilter_mod

import utils

def test_cosine_similarity_identity():
    X = [[1.0, 0.0], [0.0, 1.0]]
    Y = [[1.0, 0.0], [0.0, 1.0]]
    res = utils.cosine_similarity(X, Y)
    assert res.shape == (2, 2)
    assert np.allclose(res, np.array([[1.0, 0.0], [0.0, 1.0]]))

def test_cosine_similarity_empty_returns_empty_array():
    res = utils.cosine_similarity([], [])
    assert isinstance(res, np.ndarray)
    assert res.size == 0

def test_cosine_similarity_shape_mismatch_raises():
    X = [[1.0, 0.0, 0.0]]
    Y = [[1.0, 0.0]]
    with pytest.raises(ValueError):
        utils.cosine_similarity(X, Y)

def test_cosine_similarity_with_numpy_inputs():
    X = np.array([[1.0, 1.0]])
    Y = np.array([[1.0, 0.0]])
    res = utils.cosine_similarity(X, Y)
    expected = np.array([[1.0 / np.sqrt(2.0)]])
    assert np.allclose(res, expected)

def test_tokenize_query_with_janome_or_and_monkeypatched(monkeypatch):
    # analyzer.analyze を安定した戻り値にモンキーパッチする
    fake_tokens = [types.SimpleNamespace(surface="tok1"), types.SimpleNamespace(surface="tok2")]

    class FakeAnalyzer:
        def analyze(self, q):
            # q は無視して固定トークンを返す
            return fake_tokens

    monkeypatch.setattr(utils, "analyzer", FakeAnalyzer())

    # OR はトークン間を "OR" で連結
    assert utils.tokenize_query_with_janome("ignored", "OR") == "tok1ORtok2"
    # AND はトークン間をスペースで連結（実装上の仕様）
    assert utils.tokenize_query_with_janome("ignored", "AND") == "tok1 tok2"

def test_tokenize_query_with_janome_invalid_expr_raises():
    with pytest.raises(ValueError):
        utils.tokenize_query_with_janome("q", "XOR")


def test_ja_sentence_splitter_init_is_callable():
    pipeline = utils.ja_sentence_splitter_init()
    assert callable(pipeline)