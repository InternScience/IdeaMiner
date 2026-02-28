import os
import json
import time
import inspect
import requests
import threading
import asyncio
from collections import deque
from functools import wraps
from cachetools import LRUCache
from langchain.tools import tool
from tavily import AsyncTavilyClient


class AsyncRateLimiter:
    """
    Async rate limiter using a sliding window algorithm.
    Supports multiple concurrent agents calling simultaneously.
    """
    def __init__(self, max_requests: int, time_window: float):
        """
        Args:
            max_requests: Maximum number of requests allowed within the time window.
            time_window: Size of the time window in seconds.
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a request permit. Waits if the rate limit has been reached."""
        async with self.lock:
            now = time.monotonic()

            # Remove old request records outside the time window
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()

            # If the current request count has reached the limit, calculate wait time
            if len(self.requests) >= self.max_requests:
                # Wait until the oldest request leaves the time window
                oldest_request = self.requests[0]
                wait_time = oldest_request + self.time_window - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Re-clean expired requests
                    now = time.monotonic()
                    while self.requests and self.requests[0] <= now - self.time_window:
                        self.requests.popleft()

            # Record the current request time
            self.requests.append(now)


class MultiAPIKeyManager:
    """
    Multi-API-key manager with load balancing and per-key rate limiting.
    Each API key has its own rate limiter; the manager automatically selects
    the best available key using a round-robin strategy.
    """
    def __init__(self, api_keys: list[str], max_requests_per_key: int, time_window: float):
        """
        Args:
            api_keys: List of API keys.
            max_requests_per_key: Maximum requests per key within the time window.
            time_window: Size of the time window in seconds.
        """
        self.api_keys = api_keys
        self.rate_limiters = {
            key: AsyncRateLimiter(max_requests_per_key, time_window)
            for key in api_keys
        }
        self.current_index = 0
        self.lock = asyncio.Lock()

    async def acquire(self) -> str:
        """
        Acquire an available API key using round-robin to select the least loaded key.

        Returns:
            An available API key string.
        """
        async with self.lock:
            # Round-robin through all keys to find the first available one
            for _ in range(len(self.api_keys)):
                key = self.api_keys[self.current_index]
                rate_limiter = self.rate_limiters[key]

                # Check if this key can be used immediately (no waiting required)
                now = time.monotonic()
                # Clean up expired records
                while rate_limiter.requests and rate_limiter.requests[0] <= now - rate_limiter.time_window:
                    rate_limiter.requests.popleft()

                # If the current key still has quota, use it
                if len(rate_limiter.requests) < rate_limiter.max_requests:
                    self.current_index = (self.current_index + 1) % len(self.api_keys)
                    break

                # Try the next key
                self.current_index = (self.current_index + 1) % len(self.api_keys)
            else:
                # All keys have reached their limit; fall back to the first key and wait
                key = self.api_keys[0]
                rate_limiter = self.rate_limiters[key]

        # Execute rate limiting outside the lock to avoid holding it for long
        await rate_limiter.acquire()
        return key


def _load_tavily_api_keys() -> list[str]:
    """
    Load Tavily API keys from environment variables.

    Reads TAVILY_API_KEYS (comma-separated list) or falls back to
    TAVILY_API_KEY (single key).  Returns an empty list if neither is set
    (the error is deferred to the first actual search call).
    """
    keys_str = os.environ.get("TAVILY_API_KEYS", "")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("TAVILY_API_KEY", "").strip()
    if single:
        return [single]
    return []


TAVILY_API_KEYS = _load_tavily_api_keys()

# Global web_search API key manager.
# Rate limit: 160 requests per key per 60 seconds.
# Initialised with whatever keys are available at import time; keys are
# re-read inside web_search() if the list is empty.
web_search_api_manager: MultiAPIKeyManager | None = (
    MultiAPIKeyManager(api_keys=TAVILY_API_KEYS, max_requests_per_key=160, time_window=60)
    if TAVILY_API_KEYS else None
)


def make_hashable(obj):
    """Convert an arbitrary Python object into a hashable form."""
    if isinstance(obj, (list, tuple, set)):
        return tuple(sorted([make_hashable(i) for i in obj]))
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    else:
        return obj


def hashable_cache(func):
    """Custom LRU cache decorator that supports list/dict arguments."""
    _cache = LRUCache(maxsize=512)
    _lock = threading.Lock()

    if inspect.iscoroutinefunction(func):
        # Async function version
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = make_hashable((args, kwargs))
            with _lock:
                if key in _cache:
                    return _cache[key]
            result = await func(*args, **kwargs)
            with _lock:
                _cache[key] = result
            return result
    else:
        # Sync function version
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = make_hashable((args, kwargs))
            with _lock:
                if key in _cache:
                    return _cache[key]
            result = func(*args, **kwargs)
            with _lock:
                _cache[key] = result
            return result

    wrapper.__signature__ = inspect.signature(func)

    return wrapper


@tool(
    description="""
Search the web for relevant information to answer user queries.
Input should be a single string representing the search query.
""".strip(),
    args_schema={
        "title": "web_search",
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of domains to specifically include in the search results (optional). Maximum 300 domains.",
                "default": None,
            },
            "exclude_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of domains to specifically exclude from the search results (optional). Maximum 150 domains.",
                "default": None,
            },
            "start_date": {
                "type": "string",
                "description": """
Filters search results to include only content published on or after this date.

Use this parameter when you need to:
- Find recent developments or updates on a topic
- Exclude outdated information from search results
- Focus on content within a specific timeframe
- Combine with end_date to create a custom date range

Format must be YYYY-MM-DD (e.g., "2024-01-15" for January 15, 2024).

Examples:
- "2024-01-01" - Results from January 1, 2024 onwards
- "2023-12-25" - Results from December 25, 2023 onwards

When combined with end_date, creates a precise date range filter.

Default is None (no start date restriction).
""".strip(),
                "default": None,
            },
            "end_date": {
                "type": "string",
                "description": """
Filters search results to include only content published on or before this date.

Use this parameter when you need to:
- Exclude content published after a certain date
- Study historical information or past events
- Research how topics were covered during specific time periods
- Combine with start_date to create a custom date range

Format must be YYYY-MM-DD (e.g., "2024-03-31" for March 31, 2024).

Examples:
- "2024-03-31" - Results up to and including March 31, 2024
- "2023-12-31" - Results up to and including December 31, 2023

When combined with start_date, creates a precise date range filter.
For example: start_date="2024-01-01", end_date="2024-03-31"
returns results from Q1 2024 only.

Default is None (no end date restriction).
""".strip(),
                "default": None,
            },
        },
        "required": ["query"],
    }
)
@hashable_cache
async def web_search(
    query: str,
    include_domains: list[str] = None,
    exclude_domains: list[str] = None,
    start_date: str = None,
    end_date: str = None,
) -> str:
    try:
        global web_search_api_manager
        if web_search_api_manager is None:
            # Re-attempt key loading in case env vars were set after module import
            keys = _load_tavily_api_keys()
            if not keys:
                return json.dumps({"error": "No Tavily API key configured. Set TAVILY_API_KEYS or TAVILY_API_KEY."})
            web_search_api_manager = MultiAPIKeyManager(api_keys=keys, max_requests_per_key=160, time_window=60)

        if include_domains is not None:
            if isinstance(include_domains, str):
                include_domains = [include_domains]
            elif not isinstance(include_domains, list):
                return "'include_domains' must be a string or a list of strings."
        if exclude_domains is not None:
            if isinstance(exclude_domains, str):
                exclude_domains = [exclude_domains]
            elif not isinstance(exclude_domains, list):
                return "'exclude_domains' must be a string or a list of strings."

        # Acquire an available API key (with rate limiting)
        api_key = await web_search_api_manager.acquire()

        # Create a client with the acquired API key
        tavily_client = AsyncTavilyClient(api_key=api_key)

        response = await tavily_client.search(
            query=query,
            search_depth="advanced",
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_date=start_date,
            end_date=end_date,
            timeout=30
        )
        if isinstance(response, dict):
            for k in ["follow_up_questions", "answer", "images", "response_time", "request_id", "raw_content"]:
                response.pop(k, None)
        return json.dumps(response, indent=4, ensure_ascii=False)
    except Exception as err:
        return json.dumps({"error": str(err)}, ensure_ascii=False)


@tool(
    description="""
Search academic papers using Semantic Scholar API.
Supports filtering by venue, year range, open access, and sorting by citations.
""".strip(),
    args_schema={
        "title": "paper_search",
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for papers.",
            },
            "year_range": {
                "type": "string",
                "description": "Year range filter, e.g., '2020-2024' or '2024' (optional).",
            },
            "venue": {
                "type": "string",
                "description": "Specific venue/journal filter, e.g., 'Nature', 'Science' (optional).",
                "default": None,
            },
            "open_access": {
                "type": "boolean",
                "description": "Whether to filter for open access papers only (default: False).",
                "default": False,
            },
            "sort_by_citation": {
                "type": "boolean",
                "description": "Whether to sort by citation count (default: True).",
                "default": True,
            },
        },
        "required": ["query"],
    }
)
@hashable_cache
def paper_search(
    query: str,
    year_range: str = None,
    venue: str = None,
    open_access: bool = False,
    sort_by_citation: bool = True
) -> str:
    """
    Search academic papers using Semantic Scholar API with retry support for 429 errors
    (up to 10 attempts with a fixed 2-second delay between retries).
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    limit = 10
    offset = 0

    # Build enhanced query with optional venue filter
    enhanced_query = query
    if venue:
        enhanced_query += f' venue:"{venue}"'

    # Define the full set of fields to return
    fields = (
        "title,abstract,authors,year,publicationDate,venue,journal,"
        "citationCount,influentialCitationCount,externalIds,url,isOpenAccess,"
        "openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes"
    )

    # Build request parameters
    params = {
        "query": enhanced_query,
        "limit": limit,
        "offset": offset,
        "fields": fields
    }

    if year_range:
        params["year"] = year_range

    if open_access:
        params["openAccess"] = "true"

    # Load Semantic Scholar API key from environment (optional; increases rate limits)
    semantic_scholar_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {
        "Accept": "application/json",
        "User-Agent": "AcademicSearchBot/2.0",
    }
    if semantic_scholar_api_key:
        headers["x-api-key"] = semantic_scholar_api_key

    # Retry logic: up to 10 attempts
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)

            # On 429 with remaining retries, wait and retry
            if response.status_code == 429 and attempt < max_attempts:
                time.sleep(2)
                continue

            response.raise_for_status()

            raw_data = response.json()
            papers = raw_data.get("data", [])
            total = raw_data.get("total", 0)

            # Build structured result list
            results = []
            for p in papers:
                results.append({
                    "basic_info": {
                        "title": p.get("title"),
                        "year": p.get("year"),
                        "date": p.get("publicationDate"),
                        "venue": p.get("venue"),
                        "journal_detail": p.get("journal"),
                        "type": p.get("publicationTypes"),
                    },
                    "impact": {
                        "citations": p.get("citationCount"),
                        "influential_citations": p.get("influentialCitationCount")
                    },
                    "content": {
                        "abstract": p.get("abstract"),
                        "subjects": p.get("fieldsOfStudy") or [s.get('category') for s in p.get('s2FieldsOfStudy', [])],
                    },
                    "access": {
                        "doi": p.get("externalIds", {}).get("DOI"),
                        "s2_url": p.get("url"),
                        "pdf_url": (p.get("openAccessPdf") or {}).get("url") if p.get("isOpenAccess") else None
                    },
                    "authors": [a.get("name") for a in p.get("authors", [])]
                })

            # Sort by citation count descending
            if sort_by_citation:
                results.sort(key=lambda x: x['impact']['citations'] or 0, reverse=True)

            final_output = {
                "search_metadata": {
                    "total_found": total,
                    "count_returned": len(results),
                    "query_used": enhanced_query,
                    "attempts": attempt
                },
                "papers": results
            }

            return json.dumps(final_output, ensure_ascii=False, indent=4)

        except requests.exceptions.HTTPError as err:
            # On last attempt or non-429 error, return error info
            if attempt == max_attempts or err.response.status_code != 429:
                return json.dumps({
                    "error": f"HTTP Error: {err.response.status_code}",
                    "detail": err.response.text,
                    "attempts": attempt
                }, ensure_ascii=False)
        except Exception as e:
            # Non-HTTP errors are returned immediately
            return json.dumps({
                "error": str(e),
                "attempts": attempt
            }, ensure_ascii=False)

    # Fallback if all retries are exhausted (should not normally be reached)
    return json.dumps({
        "error": "Max retry attempts reached",
        "attempts": max_attempts
    }, ensure_ascii=False)
