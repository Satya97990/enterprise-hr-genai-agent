from src.retrieval.rag_pipeline import build_hr_agent, run_automated_check

if __name__ == "__main__":
    agent = build_hr_agent()
    run_automated_check(agent)