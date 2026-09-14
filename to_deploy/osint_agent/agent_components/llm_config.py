# LLM factory. Each model below was tuned after specific errors (token limits,
# unsupported params on certain model families) — keep the parameters as-is
# unless you hit a new error that justifies changing them.
#
# Usage from a notebook:
#     from agent_components.llm_config import build_llms
#     llms = build_llms(key_vault)
#     llms.llm_tool_selection.invoke(...)
#     llms.llm_evaluation.invoke(...)

from types import SimpleNamespace
from langchain_openai import ChatOpenAI


def build_llms(key_vault):
    """
    Build all configured ChatOpenAI instances.

    Args:
        key_vault: APIVault instance exposing .get_key(name). Must contain
                   "openai_key".

    Returns:
        SimpleNamespace with attributes:
            llm_tool_selection, llm_search_tool_payload, llm_names_variation,
            llm_translation_or_terms, llm_expert_generation, llm_hyde_generation,
            llm_url_content_summary, llm_agg_summaries, llm_evaluation.
    """

    openai_key = key_vault.get_key("openai_key")

    # Short deterministic outputs — tool selection, structured fields
    llm_tool_selection = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2,
        max_tokens=50,
        top_p=0.95,
        timeout=30,
        max_retries=3,
        api_key=openai_key
    )

    # Search query payload generation
    llm_search_tool_payload = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2,
        max_tokens=2000,
        top_p=0.95,
        timeout=120,
        max_retries=3,
        api_key=openai_key
    )

    # Name variations generation (pure extraction, near-zero temperature)
    llm_names_variation = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.0,
        max_tokens=50,
        timeout=30,
        max_retries=3,
        api_key=openai_key
    )

    # Translation and key terms extraction
    llm_translation_or_terms = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
        max_tokens=500,
        timeout=30,
        max_retries=3,
        api_key=openai_key
    )

    # Journalist persona generation
    llm_expert_generation = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2,
        max_tokens=1000,
        timeout=60,
        max_retries=3,
        api_key=openai_key
    )

    # HyDE article generation (creative variation per journalist style)
    llm_hyde_generation = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.5,
        max_tokens=500,
        timeout=60,
        max_retries=3,
        api_key=openai_key
    )

    # URL content summarization
    llm_url_content_summary = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.2,
        max_tokens=2000,
        top_p=0.95,
        timeout=120,
        max_retries=3,
        api_key=openai_key
    )

    # Evidence claims aggregation across multiple summaries.
    # max_tokens=10000 is intentional: .with_structured_output() on a RunnableBinding
    # creates a new chain from the original LLM and discards any .bind(max_tokens=...)
    # override, so the ceiling must be set in the constructor. Lowering this previously
    # triggered: LengthFinishReasonError: Could not parse response content as the
    # length limit was reached - CompletionUsage(completion_tokens=2000, prompt_tokens=11447, total_tokens=13447).
    llm_agg_summaries = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.1,
        max_tokens=10000,
        top_p=0.95,
        timeout=120,
        max_retries=3,
        api_key=openai_key
    )

    # Final evaluation and risk assessment
    llm_evaluation = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.2,
        max_tokens=3000,
        top_p=0.95,
        timeout=120,
        max_retries=3,
        api_key=openai_key
    )

    return SimpleNamespace(
        llm_tool_selection=llm_tool_selection,
        llm_search_tool_payload=llm_search_tool_payload,
        llm_names_variation=llm_names_variation,
        llm_translation_or_terms=llm_translation_or_terms,
        llm_expert_generation=llm_expert_generation,
        llm_hyde_generation=llm_hyde_generation,
        llm_url_content_summary=llm_url_content_summary,
        llm_agg_summaries=llm_agg_summaries,
        llm_evaluation=llm_evaluation,
    )
