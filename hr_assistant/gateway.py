
"""Route LLM calls through Portkey with automatic fallback."""

from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

PRIMARY_PROVIDER = "@hrpolicy"
BACKUP_PROVIDER = "@hrpolicybackup"


def _create_gateway_llm(provider: str) -> ChatOpenAI:
    headers = createHeaders(
        api_key=config.PORTKEY_API_KEY,
        provider=provider,
    )

    return ChatOpenAI(
        api_key="portkey",
        base_url=PORTKEY_GATEWAY_URL,
        model=config.LLM_MODEL_NAME,
        default_headers=headers,
    )


def get_gateway_llm():
    """Return a LangChain-compatible model with fallback."""
    primary_llm = _create_gateway_llm(PRIMARY_PROVIDER)
    backup_llm = _create_gateway_llm(BACKUP_PROVIDER)

    logger.info(
        "Configuring LLM fallback: %s -> %s",
        PRIMARY_PROVIDER,
        BACKUP_PROVIDER,
    )

    return primary_llm.with_fallbacks([backup_llm])
