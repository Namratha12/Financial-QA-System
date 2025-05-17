# main.py

import argparse
from langchain_core.messages import HumanMessage
from src.agent.pipeline import run_agent_pipeline
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def main():
    parser = argparse.ArgumentParser(description="Run the ConvFinQA RAG agent.")
    parser.add_argument("--question", type=str, required=True, help="Financial question to answer")
    args = parser.parse_args()

    question = args.question
    print(f"\n Running Agent Pipeline for:\n{question}\n")

    state = run_agent_pipeline(question)

    print("\nFinal Answer:")
    print(state.answer)
    print("\nReasoning:")
    print(state.generation)



if __name__ == "__main__":
    main()
