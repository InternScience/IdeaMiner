import requests
import json


def search_papers(
    query,
    limit=10,
    offset=0,
    fields=None,
    year_range=None,
    venue=None,
    open_access=False,
    sort_by_citation=True
):
    """
    Search academic papers using the Semantic Scholar API.

    Args:
        query: Search query string.
        limit: Maximum number of results to return (default: 10).
        offset: Pagination offset (default: 0).
        fields: Comma-separated string of fields to return. Defaults to a
                comprehensive set covering title, abstract, authors, citations, etc.
        year_range: Year filter, e.g. "2020-2024" or "2024".
        venue: Venue/journal filter, e.g. "Nature", "Science", "Cell".
        open_access: If True, restrict results to open-access papers (default: False).
        sort_by_citation: If True, sort results by citation count descending (default: True).

    Returns:
        JSON string with search metadata and a list of paper records.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    # Build enhanced query with optional venue filter
    enhanced_query = query
    if venue:
        enhanced_query += f' venue:"{venue}"'

    # Define the default set of fields to retrieve
    if fields is None:
        fields = (
            "title,abstract,authors,year,publicationDate,venue,journal,"
            "citationCount,influentialCitationCount,externalIds,url,isOpenAccess,"
            "openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes"
        )

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

    try:
        headers = {
            "Accept": "application/json",
            "User-Agent": "AcademicSearchBot/2.0"
        }

        response = requests.get(url, params=params, headers=headers, timeout=15)
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
                "query_used": enhanced_query
            },
            "papers": results
        }

        return json.dumps(final_output, ensure_ascii=False, indent=4)

    except requests.exceptions.HTTPError as err:
        return json.dumps({"error": f"HTTP Error: {err.response.status_code}", "detail": err.response.text})
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    # Example A: Search for highly cited papers about solid-state batteries in Nature
    print("--- Example A: Top-venue search ---")
    json_a = search_papers("Solid-state batteries", venue="Nature", limit=2)
    print(json_a)

    # Example B: Search for open-access CRISPR papers from 2023 to 2026
    print("\n--- Example B: Year range and open-access filter ---")
    json_b = search_papers("CRISPR", year_range="2023-2026", open_access=True, limit=2)
    print(json_b)
