"""Central LLM factory — swap providers via LLM_PROVIDER in .env."""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Load .env here so _PROVIDER is evaluated after env vars are set.
# Use override=True so project-local env vars win over stale shell exports.
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Provider: "groq" | "gemini" | "openai"
# Set LLM_PROVIDER=openai in .env to use OpenAI.
# ---------------------------------------------------------------------------
_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Groq model names
_GROQ_FAST = "meta-llama/llama-4-scout-17b-16e-instruct"    # 30K TPM, 500K TPD
_GROQ_QUALITY = "meta-llama/llama-4-scout-17b-16e-instruct" # 30K TPM, 500K TPD

# Gemini model names
_GEMINI_FAST = "gemini-2.5-flash"
_GEMINI_QUALITY = "gemini-2.5-flash"

# OpenAI model names
_OPENAI_FAST = "gpt-4o-mini"       # cheap, fast — JD parsing & review
_OPENAI_QUALITY = "gpt-4.1-mini"   # resume tailoring — better instruction following


def _with_retry(llm: BaseChatModel) -> BaseChatModel:
    """Wrap an LLM with automatic retry on rate-limit errors.

    Args:
        llm: Any LangChain chat model.

    Returns:
        The same model wrapped with retry logic.
    """
    return llm.with_retry(
        stop_after_attempt=6,
        wait_exponential_jitter=True,
    )


def get_fast_llm(temperature: float = 0) -> BaseChatModel:
    """Return the fast LLM for parsing, scoring, and reviewing.

    Args:
        temperature: Sampling temperature.

    Returns:
        Chat model with automatic retry on 429s.
    """
    if _PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=_GEMINI_FAST, temperature=temperature
        )
    elif _PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=_OPENAI_FAST, temperature=temperature)
    else:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=_GROQ_FAST, temperature=temperature)
    return _with_retry(llm)


def get_quality_llm(temperature: float = 0.2) -> BaseChatModel:
    """Return the higher-quality LLM for resume tailoring.

    Args:
        temperature: Sampling temperature.

    Returns:
        Chat model with automatic retry on 429s.
    """
    if _PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=_GEMINI_QUALITY, temperature=temperature
        )
    elif _PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=_OPENAI_QUALITY, temperature=temperature)
    else:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=_GROQ_QUALITY, temperature=temperature)
    return _with_retry(llm)
