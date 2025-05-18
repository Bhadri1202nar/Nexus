import feedparser
from urllib.parse import quote

def search_arxiv(query, max_results=5):
    print(f"[DEBUG] Searching arXiv for query: {query}")
    encoded_query = quote(f"all:{query}")
    url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}"
    print(f"[DEBUG] Constructed URL: {url}")

    feed = feedparser.parse(url)
    if not feed.entries:
        return "No research papers found."

    results = ""
    for i, entry in enumerate(feed.entries[:max_results], start=1):
        title = entry.title
        link = entry.link
        summary = entry.summary[:400] + "..." if len(entry.summary) > 400 else entry.summary
        results += f"**{i}. [{title}]({link})**\n\n{summary}\n\n"

    return results
