import asyncio
import csv
import inspect
from typing import Optional
from urllib.parse import urlparse
from pydantic import BaseModel

from dotenv import load_dotenv
from browser_use import Agent, Browser, ChatBrowserUse

load_dotenv()

BASE_SEARCH_QUERY = "('incident management' OR 'fault management') AND ('remediation' OR 'mitigation' OR 'resolution' OR 'solution') AND ('LLM' OR 'large language model' OR 'AI Agent')"

SEARCH_ENGINE = "google.com"

MAX_RESULTS = 100

GENERAL_SEARCH_EXCLUDED_SITES = [
    # "arxiv.org",
    # "aws.amazon.com",
    # "sre.google",
    # "pagerduty.com",
    # "netflixtechblog.com",
    # "atlassian.com",
    # "medium.com",
    # "dev.to",
    # "github.com",
    # "microsoft.com",
]

SEARCH_PROMPT_TEMPLATE = """
<ROLE>
You are a precise web scraping assistant specialized in extracting structured data from {search_engine} search results. You follow instructions exactly and extract only visible information without accessing individual links.
</ROLE>

<TASK>
Access {search_engine} and type this search query in the search box: {search_query}

Navigate through {search_engine} search result pages and collect EXACTLY {max_results} search results with the following information for each result:
- Complete URL (link)
- Title (clickable headline)
- Description (text snippet below the title)

Instructions:
- Start from page 1 and navigate to subsequent pages as needed to collect {max_results} results
- DO NOT click on any links, extract only data visible on {search_engine} search pages
- Include only organic search results, exclude ads, featured snippets, and other non-standard result types
- Continue navigating pages until you have collected {max_results} results or no more results are available
</TASK>

<OUTPUT_FORMAT>
Return a structured list of SearchResult objects, each containing:
- link: complete URL (e.g., "https://github.com/user/repo")
- title: search result title
- description: text snippet
</OUTPUT_FORMAT>

<QUALITY_STRATEGY>
Ensure you collect exactly {max_results} results unless {search_engine} has fewer results available. Verify each field is populated with the correct corresponding data from the search result card. Navigate systematically through pages to reach the target number.
</QUALITY_STRATEGY>
""".strip()

SPEED_OPTIMIZATION_PROMPT = """
Speed optimization instructions:
- Be extremely concise and direct in your responses
- Get to the goal as quickly as possible
- Use multi-action sequences whenever possible to reduce steps
"""


class SearchResult(BaseModel):
    link: str
    title: str
    description: str
    source: Optional[str] = ""


class SearchResults(BaseModel):
    results: list[SearchResult]


def add_source_to_results(results: list[SearchResult]) -> list[SearchResult]:
    for result in results:
        parsed = urlparse(result.link)
        domain = parsed.netloc or parsed.path

        if domain.startswith("www."):
            domain = domain[4:]

        result.source = domain

    return results


def configure_search(max_results: int) -> str:
    print(f"Searching without site filter (collecting {max_results} results)...")
    exclusions = " ".join([f"-site:{e}" for e in GENERAL_SEARCH_EXCLUDED_SITES])
    search_query = f"{BASE_SEARCH_QUERY} {exclusions}".strip()
    print(f"Query: {search_query}")
    return search_query


async def perform_search(
    search_query: str, max_results: int, llm: ChatBrowserUse, browser: Browser
) -> SearchResults:
    task = SEARCH_PROMPT_TEMPLATE.format(
        search_query=search_query,
        max_results=max_results,
        search_engine=SEARCH_ENGINE,
    )

    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        output_model_schema=SearchResults,
        flash_mode=True,
        extend_system_message=SPEED_OPTIMIZATION_PROMPT,
    )

    raw_result = await agent.run()
    final_result = raw_result.final_result()
    if not final_result:
        return SearchResults(results=[])
    parsed_results = SearchResults.model_validate_json(final_result)

    return parsed_results


def save_results_to_csv(
    results: list[SearchResult], filename: str = "./grey_literature/search_results.csv"
):
    fieldnames = ["link", "title", "description", "source"]
    rows = []
    index = {}
    try:
        with open(filename, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {k: v for k, v in row.items() if k in fieldnames}
                if "link" not in clean_row or not clean_row["link"]:
                    continue
                rows.append(clean_row)
                index[clean_row["link"]] = len(rows) - 1
    except FileNotFoundError:
        pass

    for result in results:
        data = result.model_dump()
        link = data["link"]
        if link in index:
            rows[index[link]] = data
        else:
            index[link] = len(rows)
            rows.append(data)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(results)} results to {filename}")


async def close_browser(browser):
    close = getattr(browser, "close", None)
    if close:
        result = close()
        if inspect.isawaitable(result):
            await result


async def main():
    llm = ChatBrowserUse()
    browser = None
    try:
        browser = Browser(headless=False)
        search_query = configure_search(MAX_RESULTS)
        parsed_results = await perform_search(search_query, MAX_RESULTS, llm, browser)
        results_with_source = add_source_to_results(parsed_results.results)
        save_results_to_csv(results_with_source)
    finally:
        if browser:
            await close_browser(browser)


if __name__ == "__main__":
    asyncio.run(main())
