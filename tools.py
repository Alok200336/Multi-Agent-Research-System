from langchain_core.tools import Tool, tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()


tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool

def web_search(query: str) -> str:
    """
    Perform a web search using Tavily API and return the results.
    """
  
    response = tavily.search(query=query, max_results=5)
    out = []

    for r in response["results"]:
        out.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Snippet: {r.get('content', '')[:200]}\n"
        )

    return "\n".join(out)

print(web_search.invoke("What is the capital of France?"))


@tool

def scrape_url(url: str) -> str:
    """Fetch a webpage and return its cleaned text for deeper reading."""
    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        return soup.get_text(separator=" ", strip=True)[:3000]

    except requests.RequestException as e:
        return f"Could not scrape URL: {e}"

print(scrape_url.invoke("https://indianexpress.com/article/india/rahul-gandhi-priyanka-detained-india-bloc-protest-cec-gyanesh-kumar-resignation-demand-10909708/?ref=breaking_hp"))