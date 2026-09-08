# llm_agent/tool_currency_converter.py

import time
import requests


class CurrencyConverterTool:
    """Инструмент для конвертации валют с использованием внешнего API."""

    name = "currency_converter"
    description = (
        "Конвертирует сумму из одной валюты в другую. "
        "Использует актуальные курсы валют и кеширует их на 1 час. "
        "Пример: 100 USD в EUR."
    )

    API_URL = "https://open.er-api.com/v6/latest/"

    # Кеш курсов:
    # {
    #     "USD": {
    #         "rates": {...},
    #         "timestamp": 1234567890
    #     }
    # }
    _cache = {}

    CACHE_TTL = 60 * 60  # 1 час

    def use(self, amount: float, from_currency: str, to_currency: str) -> str:
        """
        Конвертирует сумму из одной валюты в другую.

        Args:
            amount (float): Сумма для конвертации.
            from_currency (str): Исходная валюта, например "USD".
            to_currency (str): Целевая валюта, например "EUR".

        Returns:
            str: Строка с результатом или сообщением об ошибке.
        """
        try:
            from_currency = from_currency.upper()
            to_currency = to_currency.upper()

            if amount < 0:
                raise ValueError("Сумма не может быть отрицательной")

            if not from_currency or not to_currency:
                raise ValueError("Необходимо указать обе валюты")

            rates = self._get_rates(from_currency)

            if to_currency not in rates:
                raise ValueError(
                    f"Валюта '{to_currency}' не поддерживается API"
                )

            rate = rates[to_currency]
            result = amount * rate

            return (
                f"Результат конвертации: "
                f"{amount} {from_currency} = "
                f"{result:.2f} {to_currency}"
            )

        except (ValueError, requests.RequestException) as e:
            return (
                f"Ошибка: не могу выполнить конвертацию "
                f"{amount} {from_currency} -> {to_currency}. "
                f"Детали: {e}"
            )

    def _get_rates(self, base_currency: str) -> dict:
        """Получает курсы валют из кеша или внешнего API."""

        current_time = time.time()

        # Проверяем кеш
        if base_currency in self._cache:
            cached_data = self._cache[base_currency]

            if current_time - cached_data["timestamp"] < self.CACHE_TTL:
                return cached_data["rates"]

        # Если кеш устарел или отсутствует — обращаемся к API
        response = requests.get(
            f"{self.API_URL}{base_currency}",
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        if data.get("result") != "success":
            raise ValueError("API не смог вернуть курсы валют")

        rates = data["rates"]

        # Сохраняем результат в кеш
        self._cache[base_currency] = {
            "rates": rates,
            "timestamp": current_time
        }

        return rates
