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
#llm decides the tool
result=llm_with_tool.invoke("use the get_length tool to find length of following : hello how are you")#result of binded llm
#execute tool
if result.tool_calls:
   tool_call=result.tool_calls[0]

tool_name=tool_call["name"]
tool_args=tool_call['args']

tool_result=get_length.invoke(tool_args)
#send back to llm
final_response=llm_with_tool.invoke(f"lenght of text is {tool_result}")
print(final_response)
