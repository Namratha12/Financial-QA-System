# src/agent/steps.py

import re
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from src.config import config
from src.llm.openai_llm import llm
from src.llm.prompts import (
    generate_queries_prompt_template,
    reason_and_answer_prompt_template,
    extract_anwer_prompt_template,
    filter_context_prompt_template,
)
from src.vector_store.retriever import VectorRetriever
from src.vector_store.embedding_model import embedding_model
from src.common.utils import format_prompt
from src.agent.state import AgentState


def extract_question(state: AgentState) -> AgentState:
    return state.model_copy(update={"question": state.messages[-1].content})


def generate_queries(state: AgentState, _) -> AgentState:
    prompt = generate_queries_prompt_template.format(question=state.question)
    response = llm.invoke([HumanMessage(content=format_prompt(prompt))])
    queries = [q.strip() for q in response.content.split("\n") if q.strip()]
    if state.question not in queries:
        queries.append(state.question)
    return state.model_copy(update={"queries": queries})


def extract_years(text: str) -> List[str]:
    return re.findall(r"\\b(19\\d{2}|20\\d{2})\\b", text)


def retrieve_documents(state: AgentState, _) -> AgentState:
    all_docs = []
    seen_ids = set()
    years = extract_years(state.question)
    retriever = VectorRetriever()

    def search(q):
        return retriever.similarity_search(query=q, k=config.top_k_retrieval)


    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(search, q) for q in state.queries]
        for future in as_completed(futures):
            for doc in future.result():
                doc_id = doc.metadata.get("id", "")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)

    if years:
        filtered = [d for d in all_docs if any(y in d.metadata["id"] for y in years)]
        if filtered:
            all_docs = filtered

    return state.model_copy(update={"documents": all_docs})


def rerank_documents(state: AgentState, _) -> AgentState:
    from  src.llm.cohere_client import cohere_client  # Local import to avoid circular

    if config.use_ground_truth_retrieval:
        table, narrative = split_context(state.documents)
        return state.model_copy(update={
            "reranked_documents": state.documents,
            "context_table": table,
            "context_narrative": narrative,
        })

    candidates = [
        {"text": doc.page_content, "id": doc.metadata["id"]} for doc in state.documents
    ]

    response = cohere_client.rerank(
        model=config.reranker_model_name,
        query=state.question,
        documents=candidates,
        top_n=config.top_k_rerank,
    )

    reranked = [state.documents[r.index] for r in response.results]
    table, narrative = split_context(reranked)

    return state.model_copy(update={
        "reranked_documents": reranked,
        "context_table": table,
        "context_narrative": narrative,
    })


def split_context(docs: List[Document]) -> tuple[str, str]:
    tables, narratives = [], []
    for doc in docs:
        content = doc.page_content.split("passage:", 1)[-1].strip()
        if "\n\n" in content:
            table, narrative = content.split("\n\n", 1)
        else:
            table, narrative = content, ""
        tables.append(table.strip())
        narratives.append(narrative.strip())
    return "\n\n".join(tables), "\n\n".join(narratives)


def generate_answer(state: AgentState, _) -> AgentState:
    prompt = reason_and_answer_prompt_template.format(
        question=state.question,
        context_table=state.context_table,
        context_narrative=state.context_narrative,
    )

    if config.disable_llm_generation:
        result = AIMessage("[GENERATION DISABLED]")
    else:
        result = llm.invoke([HumanMessage(content=format_prompt(prompt))])

    return state.model_copy(update={
        "prompt": prompt,
        "generation": result.content,
    })


def extract_final_answer(state: AgentState) -> AgentState:
    if config.disable_llm_generation:
        return state.model_copy(update={"answer": "[GENERATION DISABLED]"})

    match = re.search(r"<ANSWER>(.*?)</ANSWER>", state.generation, re.DOTALL)
    if match:
        return state.model_copy(update={"answer": match.group(1).strip()})

    fallback_prompt = extract_anwer_prompt_template.format(
        question=state.question,
        generation=state.generation,
    )
    fallback = llm.invoke([HumanMessage(content=format_prompt(fallback_prompt))])
    return state.model_copy(update={"answer": fallback.content.strip()})


def filter_context(state: AgentState, _) -> AgentState:
    prompt = filter_context_prompt_template.format(
        question=state.question,
        documents="\n".join(doc.page_content for doc in state.reranked_documents),
    )
    result = llm.invoke([HumanMessage(content=format_prompt(prompt))])

    text = result.content.replace("<OUTPUT>", "").replace("</OUTPUT>", "")
    try:
        context, sources = re.split("sources:", text, flags=re.IGNORECASE, maxsplit=1)
        sources = [s.strip("- ") for s in sources.strip().split("\n") if s.strip()]
    except ValueError:
        context, sources = text.strip(), []

    return state.model_copy(update={"context": context.strip(), "sources": sources})
