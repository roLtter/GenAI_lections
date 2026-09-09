# llm_agent/core_v2.py

import json
import re
import requests
from typing import List, Dict, Optional
from decouple import config

from .tool_calculator import CalculatorTool
from .tool_websearch import WebSearchTool
from .tool_pdfinfo import PDFInfoTool
from .tool_currency_converter import CurrencyConverterTool


class LLMAgent:
    """
    LLM-агент, который планирует и выполняет задачи с помощью инструментов.
    Поддерживает OpenRouter и локальный Ollama.
    """

    def __init__(
        self,
        model: str = "tngtech/deepseek-r1t2-chimera",
        local: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen3.5:0.8b"
    ):
        self.local = local
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model

        if not self.local:
            self.api_key = config("OPENROUTER_API_KEY")
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = model
        else:
            self.api_key = None
            self.url = f"{self.ollama_base_url}/v1/chat/completions"
            self.model = ollama_model

        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "pdf_info": PDFInfoTool(),
            "currency_converter": CurrencyConverterTool(),
        }

        self.conversation_history = []

    def _make_api_request(
        self,
        payload: Dict,
        headers: Optional[Dict] = None
    ) -> Dict:
        if headers is None:
            headers = {}

        if not self.local:
            headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        else:
            headers["Content-Type"] = "application/json"

        try:
            response = requests.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=120
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка при запросе к API: {e}")

    def _detect_currency_request(self, query: str) -> Optional[Dict]:
        """
        Определяет запрос на конвертацию валют без обращения к LLM.
        """

        text = query.lower().strip()

        currency_aliases = {
            "USD": [
                "usd",
                "доллар",
                "доллара",
                "долларов",
                "доллары",
                "долл",
            ],
            "EUR": [
                "eur",
                "евро",
            ],
            "RUB": [
                "rub",
                "руб",
                "рубль",
                "рубля",
                "рублей",
                "рубли",
                "рубл",
            ],
            "GBP": [
                "gbp",
                "фунт",
                "фунта",
                "фунтов",
                "фунты",
            ],
            "JPY": [
                "jpy",
                "иена",
                "иены",
                "иен",
            ],
            "CNY": [
                "cny",
                "юань",
                "юаня",
                "юаней",
                "юани",
            ],
        }

        conversion_words = [
            "конверт",
            "переведи",
            "перевести",
            "перевод",
            "обмен",
            "сколько будет",
            "сколько стоит",
        ]

        if not any(word in text for word in conversion_words):
            return None

        amount_match = re.search(
            r"\b\d+(?:[.,]\d+)?\b",
            text
        )

        if not amount_match:
            return None

        amount = amount_match.group(0).replace(",", ".")

        found = []

        for currency, aliases in currency_aliases.items():
            for alias in aliases:
                match = re.search(
                    rf"(?<![а-яёa-z]){re.escape(alias)}(?![а-яёa-z])",
                    text
                )

                if match:
                    found.append((match.start(), currency))
                    break

        found.sort()

        if len(found) < 2:
            return None

        from_currency = found[0][1]
        to_currency = found[1][1]

        return {
            "action": "currency_converter",
            "input": f"{amount} {from_currency} {to_currency}"
        }

    def _extract_plan_json(self, llm_text: str) -> List[Dict]:
        """
        Извлекает plan из ответа LLM.
        """

        if not llm_text:
            return []

        text = llm_text.strip()

        text = re.sub(
            r"```json\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(r"```\s*", "", text)

        try:
            data = json.loads(text)

            if isinstance(data, dict):
                return data.get("plan", [])

        except json.JSONDecodeError:
            pass

        match = re.search(
            r'\{\s*"plan"\s*:\s*\[.*?\]\s*\}',
            text,
            flags=re.DOTALL
        )

        if match:
            try:
                data = json.loads(match.group(0))
                return data.get("plan", [])

            except json.JSONDecodeError:
                pass

        return []

    def _ask_llm_for_plan(self, query: str) -> List[Dict]:
        """
        Создаёт план действий с помощью LLM.
        """

        # Валютные запросы обрабатываем напрямую.
        currency_action = self._detect_currency_request(query)

        if currency_action:
            print(
                "> Обнаружен запрос на конвертацию валют."
            )
            print(f"> План: {currency_action}")

            return [currency_action]

        system_prompt = """
You are an AI planning assistant.

Available tools:

1. calculator
Use for mathematical calculations.
Input: mathematical expression.

2. web_search
Use for current information or real-world facts.
Input: search query in Russian.

3. pdf_info
Use for extracting information from PDF files.
Input: file path or PDF URL.

4. currency_converter
Use ONLY for currency conversion.
Input MUST be exactly:
amount FROM_CURRENCY TO_CURRENCY

Examples:
100 USD EUR
5000 RUB USD
50 EUR JPY

Return ONLY valid JSON.

If tools are needed:
{
  "plan": [
    {
      "action": "tool_name",
      "input": "tool input"
    }
  ]
}

If no tools are needed:
{
  "plan": []
}
"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "stream": False
        }

        try:
            response_data = self._make_api_request(payload)

            llm_text = response_data["choices"][0]["message"]["content"]

            print(
                f"> Ответ LLM для плана (очищенный): {llm_text}"
            )

            return self._extract_plan_json(llm_text)

        except Exception as e:
            print(
                f"Произошла ошибка при создании плана: {e}"
            )
            return []

    def _generate_final_response(self, user_query: str) -> str:
        """
        Генерирует финальный ответ на основе результатов инструментов.
        """

        conversation_log = "\n".join(
            msg["content"]
            for msg in self.conversation_history
        )

        prompt = f"""
Ответь пользователю на его вопрос кратко и информативно.

Вопрос:
{user_query}

Результаты инструментов:
{conversation_log}

Используй результаты инструментов в ответе.
"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }

        try:
            response_data = self._make_api_request(payload)

            final_text = response_data["choices"][0]["message"]["content"]

            if final_text and final_text.strip():
                return final_text.strip()

            if self.conversation_history:
                return self.conversation_history[-1]["content"]

            return "Не удалось получить ответ."

        except Exception as e:
            if self.conversation_history:
                return self.conversation_history[-1]["content"]

            return f"Ошибка при генерации ответа: {e}"

    def process_query(self, query: str) -> str:
        """
        Основной метод обработки запроса пользователя.
        """

        print(
            "Агент анализирует ваш запрос... "
            f"(Режим: {'локальный Ollama' if self.local else 'OpenRouter'})"
        )

        self.conversation_history = []

        # Шаг 1. Планирование.
        plan = self._ask_llm_for_plan(query)

        if not plan:
            print(
                "Инструменты не требуются. "
                "Генерирую ответ напрямую."
            )

            direct_prompt = (
                "Ответьте на следующий вопрос кратко и информативно:\n"
                f"{query}"
            )

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": direct_prompt
                    }
                ],
                "stream": False
            }

            try:
                response_data = self._make_api_request(payload)

                response = response_data["choices"][0]["message"]["content"]

                if response and response.strip():
                    return response.strip()

                return "Не удалось получить ответ."

            except Exception as e:
                return f"Ошибка: {e}"

        print(f"План действий: {plan}")

        # Шаг 2. Выполнение инструментов.
        for step in plan:
            tool_name = step.get("action")
            tool_input = step.get("input", "")

            if tool_name not in self.tools:
                error_msg = (
                    f"Ошибка: инструмент с именем "
                    f"'{tool_name}' не найден."
                )

                print(error_msg)

                self.conversation_history.append({
                    "role": "system",
                    "content": error_msg
                })

                continue

            print(
                f"Выполняется инструмент: '{tool_name}'"
            )

            try:
                if tool_name == "currency_converter":
                    parts = tool_input.split()

                    if len(parts) != 3:
                        raise ValueError(
                            "Неверный формат currency_converter. "
                            "Ожидается: amount FROM_CURRENCY TO_CURRENCY"
                        )

                    amount = float(parts[0])
                    from_currency = parts[1]
                    to_currency = parts[2]

                    result = self.tools[tool_name].use(
                        amount,
                        from_currency,
                        to_currency
                    )

                else:
                    result = self.tools[tool_name].use(tool_input)

                print(f"Результат: {result}")

                self.conversation_history.append({
                    "role": "system",
                    "content": (
                        f"Tool {tool_name} result: {result}"
                    )
                })

            except Exception as e:
                error_msg = (
                    f"Ошибка при выполнении инструмента "
                    f"'{tool_name}': {e}"
                )

                print(error_msg)

                self.conversation_history.append({
                    "role": "system",
                    "content": error_msg
                })

        # Для currency_converter результат уже является готовым ответом.
        # Не делаем второй дорогой запрос к Ollama.
        if any(
            step.get("action") == "currency_converter"
            for step in plan
        ):
            if self.conversation_history:
                result = self.conversation_history[-1]["content"]

                prefix = "Tool currency_converter result: "

                if result.startswith(prefix):
                    result = result[len(prefix):]

                return result

        # Для остальных инструментов просим LLM сформировать ответ.
        print("Составляю финальный ответ...")

        return self._generate_final_response(query)

    def test_ollama_connection(self) -> bool:
        """
        Проверяет соединение с локальным Ollama.
        """

        if not self.local:
            return False

        try:
            test_url = f"{self.ollama_base_url}/v1/models"
            response = requests.get(
                test_url,
                timeout=10
            )

            return response.status_code == 200

        except Exception:
            return False
