
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

print("Initializing Tavily...")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def search(query: str) -> str:
    """Search the internet for information."""
    print(f"Searching for: {query}")

    response = tavily.search(query=query, max_results=5)

    return str(response["results"])


print("Initializing LLM...")
llm = ChatOpenAI(model="gpt-5")

print("Creating agent...")
agent = create_agent(model=llm, tools=[search])

print("Agent initialized successfully!")


def main():
    print("Invoking agent...")

    result = agent.invoke({
        "messages": [
            HumanMessage(
                content="Search for 3 AI engineer jobs using LangChain in the Bay Area."
            )
        ]
    })

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
