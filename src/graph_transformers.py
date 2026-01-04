from pydantic import BaseModel, Field
from src.llm import LLMBasicModel
from yurenizer import SynonymNormalizer, NormalizerConfig
import unicodedata

class Node(BaseModel):
    """グラフのノードを表すクラス"""

    id: str = Field(description="ノードの一意な識別子")
    type: str = Field(description="ノードの型")

    model_config = {"extra": "forbid"}


class Relationship(BaseModel):
    """グラフのリレーションシップを表すクラス"""

    source: str = Field(description="リレーションシップの始点ノードID")
    target: str = Field(description="リレーションシップの終点ノードID")
    type: str = Field(description="リレーションシップの型")

    model_config = {"extra": "forbid"}


class GraphData(BaseModel):
    """グラフデータを表すクラス（StructuredOutput用）"""

    nodes: list[Node] = Field(default_factory=list, description="抽出されたノードのリスト")
    relationships: list[Relationship] = Field(
        default_factory=list, description="抽出されたリレーションシップのリスト"
    )



system_prompt = system_prompt = """
# 知識グラフ構築ガイドライン

## 1. 概要

あなたは、テキストから構造化情報を抽出し、**知識グラフ（Knowledge Graph）** を構築するために設計された高性能アルゴリズムです。
テキストに明示的に記載された情報を最大限に取り込み、**正確性を損なわない範囲で情報を網羅的に抽出**してください。
ただし、**テキストに書かれていない推測的な情報を追加してはいけません。**

* **ノード（Nodes）** は「実体（エンティティ）」または「概念（コンセプト）」を表します。
* 知識グラフの目的は、**シンプルで明快かつ広く理解できる構造**を実現することです。

---

## 2. ノードのラベル付け（Labeling Nodes）

### 一貫性の維持

* ノードラベルには、**常に基本的で汎用的な型**を使用してください。
  例：ある人物を表す場合は、常にラベルを **"人物"** とします。
  "数学者"や "科学者"などの特定分野の肩書きをラベルに使ってはいけません。

### ノードID

* **ノードIDに整数（数値）を使わないこと。**
  ノードIDは、人間が読める形の名前またはテキスト中に登場する識別子を使用します。

---

## 3. リレーションシップ（Relationships）

* **リレーションシップ** は、ノード間の関係を表します。
* リレーションシップの型も、**一般的かつ永続的なものを使用**してください。
  例：
  × `"教授になった"`
  ○ `"教授という関係"`
* 一時的・特定的な関係を避け、**時代や文脈に左右されない表現**を使います。

---

## 4. 照応解決（Coreference Resolution）

### 実体の一貫性保持

* 同じ実体が複数の名前や代名詞で言及される場合、**最も完全な識別名を使って統一**してください。
  例：
  `"桃太郎"`、`"おじいさん"`、`"おばあさん"` が同一人物を指している場合、グラフ上では常に `"桃太郎"` を使用します。

### 一貫性の目的

* 知識グラフは一貫性があり、読み手が容易に理解できる構造であることが求められます。
  実体名の統一は、その可読性と意味的整合性を維持するために不可欠です。

---
## 5. 厳格な遵守（Strict Compliance）
上記のルールには**厳密に従ってください。**
違反した場合、処理は即座に終了します。
"""

user_template = """以下のテキストからノードとリレーションシップを抽出してください。

    ノードのid、type、リレーションシップのtypeは日本語で表現してください。

テキスト:
{text}
"""


class GraphTransformers:
    def __init__(
        self,
        llm: LLMBasicModel,
        synonym_process: bool = False,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt = user_template

        self.synonym_process = synonym_process
        self.normalize_config = NormalizerConfig(
            other_language=False,
        )
        self.synonym_normalizer = SynonymNormalizer(synonym_file_path="src/file/synonyms.txt")

    def convert(self, document: str) -> GraphData:
        document = unicodedata.normalize('NFKC', document)

        if self.synonym_process:
            document = self.synonym_normalizer.normalize(document, self.normalize_config)


        response = self.llm.run(
            content=self.user_prompt.format(text=document),
            system_prompt=self.system_prompt,
            output_format=GraphData,
            temperature=0,
        )

        return response

