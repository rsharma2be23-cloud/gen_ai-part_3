from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool

from rich import print

#creating a tool
@tool
def get_length(text:str)->int:
  """returns length of charactere"""
  return len(text)

llm = ChatMistralAI(model="mistral-small-2506")

#tool binding
llm_with_tool = llm.bind_tools([get_length])

