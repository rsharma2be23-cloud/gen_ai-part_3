from langchain.tools import tool

@tool# decorator for creating the tool
def get_greeting(name:str)->str:# type hintd
  """Generate a greeting message for the user"""#docstring description of tool
  return f"Hello {name},welcome to the matrix"

result=get_greeting.invoke({"name":"Rayan"})
print(result)