from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilyResearch

search_tool=TavilyResearch(max_result=5)
llm = ChatMistralAI(model="mistral-small-2506")
#using built in tools

prompt=ChatPromptTemplate.from_template(
  """
You are a funny assistant

summarize the following news in a funny way into clear bullet points

{news}
"""
)
# this {news is the one we get below in dictionary}
chain=prompt|llm|StrOutputParser()

news_result=search_tool.invoke("latest iraq news")

result=chain.invoke({"news":news_result})

print(result)
