from gliner import GLiNER
import re
from src.rag.character.utils import ja_sentence_splitter_init
import torch

DEFAULT_LABELS = [
    "人物",
    "組織",
    "製品",
    "場所",
    "日付",
    "イベント",
]


class NamedEntityRecognition:
    model_name: str = "knowledgator/gliner-decoder-large-v1.0"

    def __init__(self, labels: list[str] = DEFAULT_LABELS):
        self.ja_sentence_splitter = ja_sentence_splitter_init()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        self.model = GLiNER.from_pretrained(self.model_name).to(device)
        self.labels = labels
        

    def zero_shot_extract(
        self, 
        text: str, 
        thr: float=0.5, 
        num_gen_sequences: int=1
    ) -> list[dict]:
        """
            Return: list[dict["start": int, "end": int, "text": str, "label": str, "score": float, "generated_labels": list[str]]]

        """
        out, offset = [], 0

        sentences = list(self.ja_sentence_splitter(text))

        
        for sentence in sentences:
            entities = self.model.predict_entities(sentence, self.labels, threshold=thr, num_gen_sequences=num_gen_sequences)
            for entity in entities:
                entity["start"] += offset
                entity["end"]   += offset
                out.append(entity)
            offset += len(sentence) + 1
        return out
    
