# src/agent/pipeline.py

from langchain_core.messages import HumanMessage
from src.agent.state import AgentState
from src.config import config
from src.agent.steps import (
    extract_question,
    generate_queries,
    retrieve_documents,
    rerank_documents,
    filter_context,
    generate_answer,
    extract_final_answer,
)

def run_agent_pipeline(question: str) -> AgentState:
    """Full RAG-based agent pipeline."""
    state = AgentState(messages=[HumanMessage(content=question)])

    state = extract_question(state)
    state = generate_queries(state, config)
    state = retrieve_documents(state, config)
    state = rerank_documents(state, config)
    state = filter_context(state, config)
    state = generate_answer(state, config)
    state = extract_final_answer(state)

    return state
