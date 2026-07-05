from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()


def main():
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    response = llm.invoke("Say hello in one short sentence.")
    print(response.content)


if __name__ == "__main__":
    main()
