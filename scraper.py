"""
scraper.py  —  Amazon Review Fetcher
======================================
Fetches real Amazon product reviews using SerpAPI.
Free tier: 100 searches/month — enough for demos.

Sign up free at: https://serpapi.com
Get your API key from: https://serpapi.com/manage-api-key
"""

import re
import requests


SERPAPI_KEY = "a12578b901e6ac761f5d92426e57c04b0522157107083e6f2a9c636ad88d739e"   # <-- paste your free key here


def extract_asin(url_or_asin: str) -> str:
    url_or_asin = url_or_asin.strip()

    # Try to extract ASIN from full Amazon URL
    match = re.search(r'/dp/([A-Z0-9]{10})', url_or_asin)
    if match:
        return match.group(1)

    # Try plain ASIN format (10 chars)
    if re.match(r'^[A-Z0-9]{10}$', url_or_asin.upper()):
        return url_or_asin.upper()

    # Accept anything if no API key (demo mode)
    if SERPAPI_KEY == "a12578b901e6ac761f5d92426e57c04b0522157107083e6f2a9c636ad88d739e":
        return url_or_asin if url_or_asin else "DEMO"

    # With real API key — try to use input as-is
    return url_or_asin if url_or_asin else None

def fetch_reviews(asin: str, num_pages: int = 2) -> dict:
    """
    Fetch Amazon product reviews for given ASIN using SerpAPI.

    Parameters:
        asin      : Amazon ASIN code (e.g. B09G9FPHY6)
        num_pages : number of review pages to fetch (each page = ~10 reviews)

    Returns:
        {
          'product_name' : str,
          'rating'       : str,
          'total_reviews': int,
          'reviews'      : [{'title': str, 'text': str, 'rating': int, 'date': str}]
          'error'        : str or None
        }
    """
    # API KEY
    if SERPAPI_KEY == "a12578b901e6ac761f5d92426e57c04b0522157107083e6f2a9c636ad88d739e":
        # Demo mode — return realistic fake data so dashboard works without API key
        return _demo_data(asin)

    all_reviews = []
    product_info = {}

    for page in range(1, num_pages + 1):
        params = {
            "engine"         : "amazon_reviews",
            "asin"           : asin,
            "api_key"        : SERPAPI_KEY,
            "amazon_domain"  : "amazon.in",   # change to amazon.com for US
            "page"           : page,
            "sort_by"        : "recent",
        }

        try:
            response = requests.get(
                "https://serpapi.com/search",
                params=params,
                timeout=15
            )
            data = response.json()

            if "error" in data:
                return {
                    'product_name' : 'Unknown',
                    'rating'       : 'N/A',
                    'total_reviews': 0,
                    'reviews'      : [],
                    'error'        : data['error']
                }

            # Extract product info from first page
            if page == 1:
                summary = data.get('reviews_results', {}).get('ratings', {})
                product_info = {
                    'product_name' : data.get('search_information', {}).get('query_displayed', asin),
                    'rating'       : str(summary.get('rating', 'N/A')),
                    'total_reviews': summary.get('total', 0),
                }

            # Extract individual reviews
            for r in data.get('reviews_results', {}).get('reviews', []):
                all_reviews.append({
                    'title' : r.get('title', ''),
                    'text'  : r.get('snippet', ''),
                    'rating': int(r.get('rating', 3)),
                    'date'  : r.get('date', ''),
                })

        except requests.exceptions.RequestException as e:
            return {
                'product_name' : 'Unknown',
                'rating'       : 'N/A',
                'total_reviews': 0,
                'reviews'      : [],
                'error'        : f"Network error: {str(e)}"
            }

    return {
        **product_info,
        'reviews': all_reviews,
        'error'  : None
    }


def _demo_data(asin: str) -> dict:
    """
    Returns realistic demo reviews when no API key is set.
    Used for testing the dashboard without spending API calls.
    """
    return {
        'product_name' : f'Demo Product ({asin})',
        'rating'       : '4.1',
        'total_reviews': 1284,
        'reviews': [
            {'title': 'Excellent watch!',
             'text' : 'This watch is absolutely brilliant. The quality is superb and the strap is very comfortable. Totally worth the price.',
             'rating': 5, 'date': 'December 2024'},
            {'title': 'Good but expensive',
             'text' : 'Nice looking watch. Build quality is decent. But the price feels a bit high for what you get.',
             'rating': 3, 'date': 'November 2024'},
            {'title': 'Worst purchase ever',
             'text' : 'Terrible quality. Stopped working after just 2 weeks. Very disappointing. Waste of money. Never buying again.',
             'rating': 1, 'date': 'November 2024'},
            {'title': 'Perfect gift',
             'text' : 'Bought this as a gift for my dad. He absolutely loves it. Looks premium and elegant.',
             'rating': 5, 'date': 'October 2024'},
            {'title': 'Battery drains fast',
             'text' : 'Watch looks good but the battery drains really quickly. Expected much better from this brand.',
             'rating': 2, 'date': 'October 2024'},
            {'title': 'Amazing product',
             'text' : 'Fantastic watch! Great display, smooth operation. Really happy with this purchase.',
             'rating': 5, 'date': 'September 2024'},
            {'title': 'Okay product',
             'text' : 'Nothing special. Average quality. Works fine but nothing to brag about.',
             'rating': 3, 'date': 'September 2024'},
            {'title': 'Broke in 1 month',
             'text' : 'Very poor build quality. The strap broke after just one month. Not recommended at all.',
             'rating': 1, 'date': 'August 2024'},
            {'title': 'Love it!',
             'text' : 'Wonderful watch. Looks exactly like the pictures. Very satisfied with the purchase.',
             'rating': 5, 'date': 'August 2024'},
            {'title': 'Decent for price',
             'text' : 'Considering the price, this is a decent watch. Not perfect but gets the job done.',
             'rating': 3, 'date': 'July 2024'},
        ],
        'error': None
    }
