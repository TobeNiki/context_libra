import os
import sys
from unittest.mock import patch, MagicMock
import types

import numpy as np
import pytest

# ここでダミーの sentence_transformers モジュールを差し込む
if "sentence_transformers" not in sys.modules:
    dummy_module = types.ModuleType("sentence_transformers")
    # とりあえず属性だけ作っておけば OK（後で patch で上書きする）
    dummy_module.SentenceTransformer = MagicMock()
    sys.modules["sentence_transformers"] = dummy_module

# `src`ディレクトリをパスに追加して、`embedding_generator`をインポート可能にする
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from embedding_generator import EmbeddingGenerator


@pytest.fixture(autouse=True)
def clear_env():
    """各テスト前後で環境変数をクリア"""
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def mock_sentence_transformer():
    """
    SentenceTransformerのコンストラクタとencodeメソッドをモック化するfixture

    戻り値:
        (mock_cls, mock_instance)
        mock_cls: SentenceTransformer クラスのモック
        mock_instance: そのインスタンス（.encode を持つ）
    """
    with patch("embedding_generator.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


def test_initialization_with_env_variables(mock_sentence_transformer):
    """環境変数から設定が読み込まれることをテスト"""
    mock_cls, _ = mock_sentence_transformer

    test_env = {
        "EMBEDDING_MODEL": "test-model",
        "EMBEDDING_PREFIX_QUERY": "query: ",
        "EMBEDDING_PREFIX_EMBEDDING": "passage: ",
    }
    with patch.dict(os.environ, test_env, clear=True):
        generator = EmbeddingGenerator()
        assert generator.model_name == "test-model"
        assert generator.prefix_query == "query: "
        assert generator.prefix_embedding == "passage: "
        mock_cls.assert_called_with("test-model")


def test_initialization_with_defaults(mock_sentence_transformer):
    """環境変数がない場合にデフォルト値が使われることをテスト"""
    mock_cls, _ = mock_sentence_transformer

    generator = EmbeddingGenerator()
    assert generator.model_name == "intfloat/multilingual-e5-large"
    assert generator.prefix_query == ""
    assert generator.prefix_embedding == ""
    mock_cls.assert_called_with("intfloat/multilingual-e5-large")


def test_add_prefix(mock_sentence_transformer):
    """_add_prefixメソッドのロジックをテスト"""
    # fixtureを受け取ることで SentenceTransformer はモック化済み
    generator = EmbeddingGenerator()

    assert generator._add_prefix("text", "prefix: ") == "prefix: text"
    assert generator._add_prefix("prefix: text", "prefix: ") == "prefix: text"
    assert generator._add_prefix("text", "") == "text"
    assert generator._add_prefix("TEXT", "prefix: ") == "prefix: TEXT"


def test_generate_embedding_with_prefix(mock_sentence_transformer):
    """generate_embeddingが正しいプレフィックスを使用することをテスト"""
    _, mock_instance = mock_sentence_transformer

    test_env = {"EMBEDDING_PREFIX_EMBEDDING": "passage: "}
    with patch.dict(os.environ, test_env, clear=True):
        generator = EmbeddingGenerator()
        generator.generate_embedding("my text")
        mock_instance.encode.assert_called_with("passage: my text")


def test_generate_embeddings_with_prefix(mock_sentence_transformer):
    """generate_embeddingsが正しいプレフィックスを使用することをテスト"""
    _, mock_instance = mock_sentence_transformer

    test_env = {"EMBEDDING_PREFIX_EMBEDDING": "passage: "}
    with patch.dict(os.environ, test_env, clear=True):
        generator = EmbeddingGenerator()
        generator.generate_embeddings(["text1", "text2"])
        mock_instance.encode.assert_called_with(["passage: text1", "passage: text2"])


def test_generate_query_embedding_with_prefix(mock_sentence_transformer):
    """generate_query_embeddingが正しいプレフィックスを使用することをテスト"""
    _, mock_instance = mock_sentence_transformer

    test_env = {"EMBEDDING_PREFIX_QUERY": "query: "}
    with patch.dict(os.environ, test_env, clear=True):
        generator = EmbeddingGenerator()
        generator.generate_search_embedding("my query")
        mock_instance.encode.assert_called_with("query: my query")
