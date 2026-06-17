import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient

# ---------------- WEATHER TOOL ----------------

API_KEY = os.getenv("WEATHER_API_KEY")

@tool
def get_weather(city: str) -> str:
    """Get current weather of city"""

    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"

    response = requests.get(url)
    data = response.json()

    if "error" in data:
        return f"Error: {data['error']['message']}"

    temp = data["current"]["temp_c"]
    desc = data["current"]["condition"]["text"]

    return f"Weather in {city}: {desc}, {temp}°C"


# ---------------- NEWS TOOL ----------------

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
        content = article.get("content", "")

        news_list.append(
            f"Title: {title}\n{content}"
        )

    return "\n\n".join(news_list)


# ---------------- LLM ----------------

llm = ChatMistralAI(model="mistral-small-2506")

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tools = llm.bind_tools([
    get_weather,
    get_news
])

# ---------------- STREAMLIT UI ----------------

st.set_page_config(
    page_title="City Intelligence System",
    page_icon="🌍"
)

st.title("🌍 City Intelligence System")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

user_input = st.chat_input(
    "Ask about weather or city news..."
)

if user_input:

    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    while True:

        result = llm_with_tools.invoke(
            st.session_state.messages
        )

        st.session_state.messages.append(result)

        if result.tool_calls:

            for tool_call in result.tool_calls:

                tool_name = tool_call["name"]

                with st.chat_message("assistant"):
                    st.info(
                        f"Using tool: {tool_name}"
                    )

                tool_result = tools[tool_name].invoke(
                    tool_call["args"]
                )

                st.session_state.messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )

            continue

        else:

            with st.chat_message("assistant"):
                st.write(result.content)

            break