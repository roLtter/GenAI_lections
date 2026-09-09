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
    Поддерживает как OpenRouter API, так и локальный Ollama.
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
        Надёжно определяет простые запросы на конвертацию валют.

        Это fallback для маленьких локальных моделей, которые могут
        неправильно выбрать инструмент или вернуть невалидный JSON.
        """

        text = query.lower().strip()

        currency_patterns = {
            "usd": [
                r"\bдоллар(?:ов|а)?\b",
                r"\bдолл(?:ар)?\.?\b",
                r"\busd\b",
                r"\$\b",
            ],
            "eur": [
                r"\bевро\b",
                r"\beur\b",
                r"€",
            ],
            "rub": [
                r"\bруб(?:ль|ля|лей)?\b",
                r"\bрубл(?:ей|я|ь)?\b",
                r"\brub\b",
                r"\bроссийск(?:их|ий)\s+руб",
            ],
            "gbp": [
                r"\bфунт(?:ов|а)?\b",
                r"\bgbp\b",
            ],
            "jpy": [
                r"\bиен(?:ы|а)?\b",
                r"\bjpy\b",
            ],
            "cny": [
                r"\bюан(?:ей|я)?\b",
                r"\bcny\b",
            ],
        }

        # Не считаем обычное упоминание валюты конвертацией.
        conversion_words = (
            "конверт",
            "перевод",
            "переведи",
            "перевести",
            "сколько будет",
            "сколько стоит",
            "обмен",
            "в ",
        )

        if not any(word in text for word in conversion_words):
            return None

        detected_currencies = []

        for currency, patterns in currency_patterns.items():
            if any(re.search(pattern, text) for pattern in patterns):
                detected_currencies.append(currency)

        # Для конвертации нужны две разные валюты.
        if len(detected_currencies) < 2:
            return None

        # Ищем число.
        amount_match = re.search(
            r"(?<!\w)(\d+(?:[.,]\d+)?)(?!\w)",
            text
        )

        if not amount_match:
            return None

        amount = float(amount_match.group(1).replace(",", "."))

        # Определяем направление по фразе "... из X в Y"
        explicit_from_to = re.search(
            r"(?:из|from)\s+([a-zа-яё]+).*?"
            r"(?:в|во|to)\s+([a-zа-яё]+)",
            text
        )

        if explicit_from_to:
            from_text = explicit_from_to.group(1)
            to_text = explicit_from_to.group(2)

            def find_currency(value):
                for currency, patterns in currency_patterns.items():
                    if any(
                        re.search(pattern, value)
                        for pattern in patterns
                    ):
                        return currency
                return None

            from_currency = find_currency(from_text)
            to_currency = find_currency(to_text)

            if from_currency and to_currency:
                return {
                    "action": "currency_converter",
                    "input": (
                        f"{amount} "
                        f"{from_currency.upper()} "
                        f"{to_currency.upper()}"
                    )
                }

        # Если направление явно не написано, определяем порядок
        # по положению валют в исходном тексте.
        positions = []

        for currency, patterns in currency_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    positions.append((match.start(), currency))
                    break

        positions.sort()

        if len(positions) >= 2:
            from_currency = positions[0][1]
            to_currency = positions[1][1]

            return {
                "action": "currency_converter",
                "input": (
                    f"{amount} "
                    f"{from_currency.upper()} "
                    f"{to_currency.upper()}"
                )
            }

        return None

    def _extract_plan_json(self, llm_text: str) -> List[Dict]:
        """
        Пытается извлечь plan из ответа LLM даже если модель
        добавила markdown, лишний текст или несколько JSON-блоков.
        """

        if not llm_text:
            return []

        text = llm_text.strip()

        # Убираем markdown fences.
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*", "", text)

        # Сначала пробуем весь ответ.
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data.get("plan", [])
        except json.JSONDecodeError:
            pass

        # Ищем объект, содержащий "plan".
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

        Для валютных запросов сначала используется детерминированное
        определение, чтобы маленькие локальные модели не ломали
        выполнение currency_converter.
        """

        # Надёжный путь для валютных запросов.
        currency_action = self._detect_currency_request(query)

        if currency_action:
            print(
                "> Определён запрос на конвертацию валют "
                "без участия LLM-планировщика."
            )
            print(f"> План: [{currency_action}]")
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

IMPORTANT:
- Return ONLY valid JSON.
- Do not use Markdown.
- Do not add explanations.
- Do not add text before or after JSON.

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

Вопрос пользователя:
{user_query}

Результаты работы инструментов:
{conversation_log}

Используй результаты инструментов в ответе.
Если результат содержит готовый результат конвертации валюты,
обязательно укажи его пользователю.
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

            # Защита от пустого ответа модели.
            if final_text and final_text.strip():
                return final_text.strip()

            # Если модель вернула пустоту, возвращаем результат инструмента.
            if self.conversation_history:
                return self.conversation_history[-1]["content"]

            return "Не удалось получить ответ."

        except Exception as e:
            # Даже при ошибке финальной генерации не теряем
            # результат инструмента.
            if self.conversation_history:
                return self.conversation_history[-1]["content"]

            return (
                f"Ошибка при генерации финального ответа. "
                f"Детали: {e}"
            )

    def process_query(self, query: str) -> str:
        """
        Основной метод обработки запроса пользователя.
        """

        print(
            "Агент анализирует ваш запрос... "
            f"(Режим: {'локальный Ollama' if self.local else 'OpenRouter'})"
        )

        # Очищаем историю предыдущего запроса.
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

        # Шаг 2. Выполнение плана.
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
                            "Для currency_converter нужен формат: "
                            "amount FROM_CURRENCY TO_CURRENCY"
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

                print(f"Результат: {result}...")

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

        # Шаг 3. Финальный ответ.
                # Для конвертации валют результат инструмента уже является
        # готовым ответом. Не делаем дополнительный запрос к LLM.
        if any(
            step.get("action") == "currency_converter"
            for step in plan
        ):
            if self.conversation_history:
                result = self.conversation_history[-1]["content"]

                # Убираем технический префикс "Tool currency_converter result: "
                prefix = "Tool currency_converter result: "
                if result.startswith(prefix):
                    result = result[len(prefix):]

                return result

        # Для остальных инструментов используем LLM
        # для формирования финального ответа.
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
            response = requests.get(test_url, timeout=10)
            return response.status_code == 200

        except Exception:
            return False
