from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch


# llm = ChatOpenAI(model="gpt-5")
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2
    )
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="search for 3 SDET job for ai engineer which involve GenAI test automation using lanchain in bay area, California and list their details")})
    print(result)

if __name__ == "__main__":
    main()
