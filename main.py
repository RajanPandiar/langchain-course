from dotenv import load_dotenv
from typing import List

from pydantic import BaseModel, Field

class Source(BaseModel):
    """Schema for the source used by the agent"""
    url: str = Field(description="The url of the source")
    title: str = Field(description="The title of the source")
    description: str = Field(description="The description of the source")

class AgentResponse(BaseModel):
    """Schema for the response used by the agent"""
    sources: List[Source] = Field(default_factory=list, description="The sources used to answer the query")
    answer: str = Field(description="The agents' answer to the query")

load_dotenv()
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

llm = ChatAnthropic(temperature=0.0, model="claude-sonnet-4-5-20250929")
tools = [TavilySearch(max_results=3)]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


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
