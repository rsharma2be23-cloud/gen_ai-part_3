from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print

API_KEY = os.getenv("WEATHER_API_KEY")

# Weather tool
@tool
def get_weather(city: str) -> str:
    """Get current weather of city"""

    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"

    response = requests.get(url)
    data = response.json()

    print("DEBUG:", data)

    if "error" in data:
        return f"Error: {data['error']['message']}"

    temp = data["current"]["temp_c"]
    desc = data["current"]["condition"]["text"]

    return f"Weather in {city}: {desc}, {temp}°C"


# Tavily news tool
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for article in results:
        title = article.get("title", "No Title")
        news_list.append(f"- {title}")

    return f"Latest news in {city}:\n" + "\n".join(news_list)
print(get_news.invoke("patiala"))

llm = ChatMistralAI(model="mistral-small-2506")
tool={
    "get_weather":get_weather,
    "get_news": get_news
}

llm_with_tools=llm.bind_tools([get_weather,get_news])
#Agent Loop
messages=[]

print("City intelligence system")
print("type exit to quit")

while True:
    user_input=input("You: ")
    if user_input.lower()=="exit":
        break
    messages.append(HumanMessage(content=user_input))#just langchains way of saying this message came from a human
    while True:
        result=llm_with_tools.invoke(messages)

        messages.append(HumanMessage(content=user_input))

        while True:
            result=llm_with_tools.invoke(messages)

            messages.append(result)

            if result.tool_calls:
                for tool_call in result.tool_calls:
                    tool_name=tool_call['name']

                    #human in the loop


