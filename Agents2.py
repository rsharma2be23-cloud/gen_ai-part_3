from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print

# Read Weather API key from .env
API_KEY = os.getenv("WEATHER_API_KEY")

# ---------------- WEATHER TOOL ----------------

@tool
def get_weather(city: str) -> str:
    """Get current weather of city"""

    # Build Weather API URL
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"

    # Send request to Weather API
    response = requests.get(url)

    # Convert JSON response to Python dictionary
    data = response.json()

    # Print raw API response for debugging
    print("DEBUG:", data)

    # Handle API errors
    if "error" in data:
        return f"Error: {data['error']['message']}"

    # Extract weather information
    temp = data["current"]["temp_c"]
    desc = data["current"]["condition"]["text"]

    # Return final weather response
    return f"Weather in {city}: {desc}, {temp}°C"


# ---------------- NEWS TOOL ----------------

# Create Tavily client
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""

    # Search latest news
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    # Extract results safely
    results = response.get("results", [])

    # No results found
    if not results:
        return f"No news found for {city}"

    news_list = []

    # Extract headlines
    for article in results:
        title = article.get("title", "No Title")
        news_list.append(f"- {title}")

    # Return all headlines
    return f"Latest news in {city}:\n" + "\n".join(news_list)


# ---------------- LLM ----------------

# Initialize Mistral model
llm = ChatMistralAI(model="mistral-small-2506")

# Map tool names to actual functions
tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

# Bind tools to the LLM
llm_with_tools = llm.bind_tools([
    get_weather,
    get_news
])

# ---------------- AGENT LOOP ----------------

# Stores full conversation history
messages = []

print("City Intelligence System")
print("Type exit to quit")

while True:

    # Get user query
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Add user's message to history
    messages.append(
        HumanMessage(content=user_input)
    )

    while True:

        # LLM decides what to do next
        result = llm_with_tools.invoke(messages)

        # Store AI response/tool call
        messages.append(result)

        # Check if tool is required
        if result.tool_calls:

            # Handle all requested tools
            for tool_call in result.tool_calls:

                # Tool selected by LLM
                tool_name = tool_call["name"]

                # Human approval before tool execution
                confirm = input(
                    f"Agent wants to call {tool_name}. Approve? (yes/no): "
                )

                if confirm.lower() == "no":
                    print("Tool call denied.")
                    break

                # Execute tool
                tool_result = tools[tool_name].invoke(
                    tool_call["args"]
                )

                # Add tool output to conversation
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )

            # Let LLM continue after seeing tool result
            continue

        else:
            # Final answer from LLM
            print(result.content)
            break

