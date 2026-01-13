# graph_of_thoughts/language_models/gemini.py
import logging
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from .abstract_language_model import AbstractLanguageModel


class GeminiModel(AbstractLanguageModel):
    """
    Triển khai mô hình Google Gemini 3 sử dụng SDK google-genai mới nhất.
    Hỗ trợ Gemini 3 Pro/Flash với tính năng thinking_level.
    """

    def __init__(self, config_path: str = "", model_name: str = "gemini-3-pro-preview", cache: bool = False) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.load_config(config_path)

        # Lấy API key từ config hoặc biến môi trường
        self.api_key = self.config[model_name].get("api_key", "")
        if not self.api_key:
            raise ValueError(f"API Key for {model_name} not found in config")

        # Khởi tạo client Google GenAI
        self.client = genai.Client(api_key=self.api_key)

        # Cấu hình Thinking Level (dành cho Gemini 3)
        # Các mức: "low", "high" (Gemini 3 Pro) hoặc "minimal", "medium" (Gemini 3 Flash)
        self.thinking_level = self.config[model_name].get("thinking_level", "high")
        self.temperature = self.config[model_name].get("temperature", 1.0)  # Gemini 3 khuyến nghị temp ~1.0

    def query(self, thought: str, n: int = 1) -> List[str]:
        """
        Gửi yêu cầu (query) đến Gemini và trả về danh sách câu trả lời.
        """
        responses = []

        # Cấu hình cho lần gọi này
        generate_config = types.GenerateContentConfig(
            temperature=self.temperature,
            candidate_count=1,  # Gemini thường trả về 1 candidate mỗi lần gọi, ta sẽ lặp để lấy n
            thinking_config=types.ThinkingConfig(
                thinking_level=self.thinking_level
            ) if "gemini-3" in self.model_name else None
        )

        for _ in range(n):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=thought,
                    config=generate_config
                )

                if response.text:
                    responses.append(response.text)
                else:
                    logging.warning(f"Gemini {self.model_name} trả về phản hồi rỗng.")
                    responses.append("")

            except Exception as e:
                logging.error(f"Lỗi khi gọi Gemini {self.model_name}: {e}")
                responses.append("")

        return responses

    def get_response(self, thought: str, n: int = 1) -> List[str]:
        """
        Alias cho phương thức query để tương thích ngược nếu cần.
        """
        return self.query(thought, n)