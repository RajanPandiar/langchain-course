import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def configure_langsmith() -> bool:
    """Enable LangSmith tracing when an API key is configured."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return False

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "langchain-course")
    return True


def main():
    tracing_enabled = configure_langsmith()
    if tracing_enabled:
        print(f"LangSmith tracing enabled (project: {os.environ['LANGSMITH_PROJECT']})")

    llm = ChatAnthropic(temperature=0.0, model="claude-sonnet-4-5-20250929")
    # llm = ChatOllama(model="gpt-oss:latest")

    information = """Narendra Damodardas Modi[a] (born 17 September 1950) is an Indian politician who has served as the prime minister of India since 26 May 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the Member of Parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindutva paramilitary volunteer organisation. He is India's third-longest-serving prime minister, and the longest-serving prime minister outside the Indian National Congress.[b]
Modi was born and raised in Vadnagar, where he completed his secondary education. He was introduced to the RSS at the age of eight, becoming a full-time worker for the organisation in Gujarat in 1971. The RSS assigned him to the BJP in 1985, and he rose through the party hierarchy, becoming general secretary in 1998.[c] In 2001, Modi was appointed chief minister of Gujarat and elected to the legislative assembly soon after. His administration is considered complicit in the 2002 Gujarat violence[d] in which over 1,000 people, mostly Muslims, were killed, with many others raped or mutilated.[e] An investigation authorised by the Supreme Court found no evidence to prosecute Modi.[f] While his policies as chief minister were credited for encouraging economic growth, his administration was criticised for failing to significantly improve health, poverty and education indices in the state.[g]"""

    summary_template = """
    Given  the information {information} about a person, i want you to create:
    1. A short summary of the information.
    2. Any two interesting facts about the person.
    """
    summary_prompt = PromptTemplate(
        template=summary_template,
        input_variables=["information"],
    )
    summary_chain = summary_prompt | llm
    summary = summary_chain.invoke(
        {"information": information},
        config={"run_name": "person_summary"},
    )
    print(summary.content)


if __name__ == "__main__":
    main()
