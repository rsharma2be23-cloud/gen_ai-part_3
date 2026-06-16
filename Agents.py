# Tavily news tool
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""

    # Search news using Tavily
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    # Extract results from response
    results = response.get("results", [])

    # Handle case when no news is found
    if not results:
        return f"No news found for {city}"

    # Store news headlines
    news_list = []

    for article in results:
        title = article.get("title", "No Title")
        news_list.append(f"- {title}")

    # Return top news headlines
    return f"Latest news in {city}:\n" + "\n".join(news_list)