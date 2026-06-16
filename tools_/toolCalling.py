
#revise later
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from rich import print

# 1. Create a custom tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text"""
    return len(text)

# Store tools in a dictionary for easy lookup
tools = {
    "get_text_length": get_text_length
}

# 2. Initialize the LLM
llm = ChatMistralAI(model="mistral-small-2506")

# 3. Bind tool to the LLM
# This lets the LLM know the tool exists
llm_with_tool = llm.bind_tools([get_text_length])

# 4. Take user input
prompt = input("You: ")

# Message history
messages = []

# Add user message
messages.append(HumanMessage(content=prompt))

# 5. First LLM call
# LLM decides whether a tool is needed
result = llm_with_tool.invoke(messages)

# Add AI response (which may contain a tool call)
messages.append(result)

# 6. Execute the tool if requested by the LLM
if result.tool_calls:

    # Get tool name and arguments
    tool_name = result.tool_calls[0]["name"]
    tool_args = result.tool_calls[0]["args"]

    # Run the tool
    tool_result = tools[tool_name].invoke(tool_args)

    # Add tool result to message history
    messages.append(tool_result)

    # 7. Second LLM call
    # LLM uses tool output to generate final answer
    final_response = llm_with_tool.invoke(messages)

    print("\n[bold green]Final Answer:[/bold green]")
    print(final_response.content)

else:
    # If no tool was needed
    print(result.content)