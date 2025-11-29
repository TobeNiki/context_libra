from pydantic import BaseModel
from typing import Any

from abc import ABC, abstractmethod
class LLMBasicModel(ABC):
    @abstractmethod
    def run(self, content: str, system_prompt: str, output_format: Any, temperature: int = 0) -> Any | None:
        pass


from openai import OpenAI

class OpenAILLM(LLMBasicModel):
    
    def __init__(self, model: str = 'gpt-4o-mini'):
        self.client = OpenAI()
        self.model = model

    def run(self, content: str, system_prompt: str, output_format: BaseModel, temperature: int = 0) -> Any | None:
        response = self.client.responses.parse(
            model=self.model,
            temperature=temperature,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            #extra_body={"reasoning_effort": "minimal"},
            text_format=output_format,
        )
        return response.output_parsed
    
from ollama import chat
class ollamaLLM(LLMBasicModel):
    def __init__(self, model: str = 'gpt-4o-mini'):
        self.model = model

    def run(self, content: str, system_prompt: str, output_format: BaseModel, temperature: int = 0) -> Any | None:
        response = chat(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            format=output_format.model_json_schema(),
        )
        return output_format.model_validate_json(response.message.content)
