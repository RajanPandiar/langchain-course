from dotenv import load_dotenv

load_dotenv()
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

llm = ChatAnthropic(temperature=0.0, model="claude-sonnet-4-5-20250929")
tools = [TavilySearch(max_results=3)]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for 3 job posting for a senior software engineer in the Banglore India?"
                )
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
