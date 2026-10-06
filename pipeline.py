from langchain_core.output_parsers import StrOutputParser

from agents import (
    search_agent,
    reader_agent,
    writer_chain,
    critic_chain,
)


def run_research_pipeline(topic: str) -> dict:
    """Search, read sources, write a report, and review it."""
    topic = topic.strip()
    if not topic:
        raise ValueError("Please enter a research topic.")

    state = {"topic": topic}
    parser = StrOutputParser()

    # 1. Search for relevant sources
    print("\n🔍 STEP 1: Searching...\n", flush=True)

    search_result = search_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Research this topic: {state['topic']}\n"
                    "Find relevant sources using web_search. "
                    "Include findings and exact source URLs.",
                )
            ]
        },
        config={"recursion_limit": 20},
    )

    state["search_results"] = parser.invoke(
        search_result["messages"][-1]
    )
    print(state["search_results"], flush=True)

    # 2. Read the sources
    print("\n📖 STEP 2: Reading sources...\n", flush=True)

    reader_result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Research topic: {state['topic']}\n\n"
                    f"Search findings:\n{state['search_results']}\n\n"
                    "Use scrape_url to read up to three relevant URLs "
                    "from these findings. Summarize the evidence and "
                    "keep source URLs beside each finding.",
                )
            ]
        },
        config={"recursion_limit": 20},
    )

    state["reader_notes"] = parser.invoke(
        reader_result["messages"][-1]
    )
    print(state["reader_notes"], flush=True)

    # Combine evidence for the writer and critic
    state["research"] = (
        f"SEARCH FINDINGS:\n{state['search_results']}\n\n"
        f"READER NOTES:\n{state['reader_notes']}"
    )

    # 3. Write the report
    print("\n✍️ STEP 3: Writing report...\n", flush=True)

    state["report"] = writer_chain.invoke({
        "topic": state["topic"],
        "research": state["research"],
    })
    print(state["report"], flush=True)

    # 4. Review the report
    print("\n📝 STEP 4: Reviewing report...\n", flush=True)

    state["critique"] = critic_chain.invoke({
        "topic": state["topic"],
        "research": state["research"],
        "report": state["report"],
    })
    print(state["critique"], flush=True)

    return state


if __name__ == "__main__":
    topic = input("Enter your research topic: ")
    result = run_research_pipeline(topic)