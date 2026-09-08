import pytest
from unittest.mock import Mock, patch

from llm_agent.tool_currency_converter import CurrencyConverterTool


@pytest.fixture(autouse=True)
def clear_cache():
    CurrencyConverterTool._cache.clear()


# =====================================================================
# ЮНИТ-ТЕСТЫ
# =====================================================================

def test_same_currency():
    tool = CurrencyConverterTool()

    mock_response = Mock()
    mock_response.json.return_value = {
        "result": "success",
        "rates": {
            "USD": 1.0
        }
    }
    mock_response.raise_for_status.return_value = None

    with patch(
        "llm_agent.tool_currency_converter.requests.get",
        return_value=mock_response
    ):
        result = tool.use(100, "USD", "USD")

    assert "100 USD = 100.00 USD" in result


def test_invalid_amount():
    tool = CurrencyConverterTool()

    result = tool.use(-100, "USD", "EUR")

    assert "Ошибка" in result
    assert "Сумма не может быть отрицательной" in result


def test_currency_conversion():
    tool = CurrencyConverterTool()

    mock_response = Mock()
    mock_response.json.return_value = {
        "result": "success",
        "rates": {
            "EUR": 0.85
        }
    }
    mock_response.raise_for_status.return_value = None

    with patch(
        "llm_agent.tool_currency_converter.requests.get",
        return_value=mock_response
    ):
        result = tool.use(100, "USD", "EUR")

    assert "100 USD = 85.00 EUR" in result


def test_cache():
    tool = CurrencyConverterTool()

    mock_response = Mock()
    mock_response.json.return_value = {
        "result": "success",
        "rates": {
            "EUR": 0.85
        }
    }
    mock_response.raise_for_status.return_value = None

    with patch(
        "llm_agent.tool_currency_converter.requests.get",
        return_value=mock_response
    ) as mock_get:

        result1 = tool.use(100, "USD", "EUR")
        result2 = tool.use(200, "USD", "EUR")

        assert "85.00 EUR" in result1
        assert "170.00 EUR" in result2

        mock_get.assert_called_once()


# =====================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# =====================================================================

@pytest.mark.integration
def test_currency_conversion_live():
    from llm_agent.core_v2 import LLMAgent

    agent = LLMAgent(
        local=True,
        ollama_model="deepseek-r1:14b"
    )

    query = "Сколько будет 100 долларов США в евро?"
    response = agent.process_query(query)

    assert "EUR" in response or "евро" in response
    assert "USD" in response or "доллар" in response


@pytest.mark.integration
def test_rub_to_usd_conversion_live():
    from llm_agent.core_v2 import LLMAgent

    agent = LLMAgent(
        local=True,
        ollama_model="deepseek-r1:14b"
    )

    query = "Конвертируй 1000 рублей в доллары США."
    response = agent.process_query(query)

    assert "USD" in response or "доллар" in response
    assert "RUB" in response or "руб" in response
