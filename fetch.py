"""Functions for fetching and caching puzzle data from cluesbysam.com."""

import requests
import json
import json5
import os
import hashlib
import re
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Tuple, List


def extract_clues_from_js_url(js_url: str) -> List[dict]:
    """Fetch and extract clues data from a JavaScript file URL.

    Args:
        js_url: Full URL to the JavaScript file containing clues data

    Returns:
        List of clue dictionaries

    Raises:
        ValueError: If clues data pattern is not found in JS file
    """
    print("Fetching clues data from " + js_url)

    # Fetch the JS file
    js_response = requests.get(js_url)
    js_response.raise_for_status()

    # Search for the clues data pattern in the JS content
    js_content = js_response.text

    # Look for array of objects with criminal, profession, name, gender fields
    # Pattern matches: =[{criminal:!0,profession:"...",name:"...",gender:"...",...},...],nextVar=
    # Stop at ] followed by comma and next variable assignment
    pattern = r'=(\[\{.*?criminal:.*?profession:.*?name:.*?gender:.*?\}]),'

    match = re.search(pattern, js_content, re.DOTALL)

    if not match:
        raise ValueError("Could not find clues data in JS file")

    # Extract the array part
    clues_text = match.group(1)

    # Try to convert JS object to valid JSON
    # Replace single quotes with double quotes, handle JS boolean values
    json_text = clues_text.replace('!0', 'true').replace('!1', 'false')
    with open('json_text.txt', 'w') as f:
        f.write(json_text)

    clues_data = json5.loads(json_text)

    return clues_data


def fetch_clues_from_website(url=None) -> Tuple[List[dict], str]:
    """Fetch clues data from cluesbysam.com (or custom URL) and save to dated JSON file.

    Args:
        url: Optional custom URL to fetch from. If None, uses default cluesbysam.com

    Returns:
        Tuple of (clues_data list, cache_filename)
    """
    base_url = url or "https://cluesbysam.com"
    print(f"{base_url=}")

    # Fetch the main page
    response = requests.get(base_url)
    response.raise_for_status()

    # Parse HTML to find the script tag
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find script tag in head that references index JS file
    script_tags = soup.find('head').find_all('script', src=True)
    index_script_src = None

    print(f"{script_tags=}")

    for script in script_tags:
        src = script.get('src')
        if src and 'index-' in src and src.endswith('.js'):
            index_script_src = src
            break

    if not index_script_src:
        raise ValueError("Could not find index JS file in HTML")

    # Construct full URL for the JS file
    print(f"Joining {base_url=} and {index_script_src=}")
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    js_url = urljoin(base_url, index_script_src)

    # Fetch and extract clues data from the JS file
    clues_data = extract_clues_from_js_url(js_url)

    # Create filename in .cache folder
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)

    if url:
        # Use MD5 hash of URL for filename
        url_hash = hashlib.md5(url.encode()).hexdigest()
        filename = os.path.join(cache_dir, f"{url_hash}.json")
    else:
        # Use today's date for default URL
        today = datetime.now().strftime("%Y%m%d")
        filename = os.path.join(cache_dir, f"{today}.json")

    # Save to file
    with open(filename, 'w') as f:
        json.dump(clues_data, f, indent=2)

    print(f"Fetched {len(clues_data)} clues and saved to {filename}")
    print("Preview of data:")
    print(json.dumps(clues_data[:2], indent=2))  # Show first 2 items

    return clues_data, filename


def get_cache_filename(puzzle_value: str) -> str:
    """Get the cache filename for a puzzle value (date or URL).

    Args:
        puzzle_value: Either a date string (YYYYMMDD) or a URL

    Returns:
        Path to cache file
    """
    cache_dir = ".cache"

    if puzzle_value.startswith(('http://', 'https://')):
        # URL: use MD5 hash
        url_hash = hashlib.md5(puzzle_value.encode()).hexdigest()
        return os.path.join(cache_dir, f"{url_hash}.json")
    else:
        # Date string: use date directly
        return os.path.join(cache_dir, f"{puzzle_value}.json")


def load_puzzle_from_cache(puzzle_value: str) -> Tuple[List[dict] | None, str]:
    """Load puzzle data from cache if it exists.

    Args:
        puzzle_value: Either a date string (YYYYMMDD) or a URL

    Returns:
        Tuple of (clues_data or None, cache_filename)
    """
    cache_filename = get_cache_filename(puzzle_value)

    if os.path.exists(cache_filename):
        with open(cache_filename, 'r') as f:
            return json.load(f), cache_filename

    return None, cache_filename


def fetch_and_cache_puzzle(puzzle_value: str) -> Tuple[List[dict], str]:
    """Fetch puzzle data and cache it.

    Args:
        puzzle_value: Either a date string (YYYYMMDD) or a URL

    Returns:
        Tuple of (clues_data, cache_filename)
    """
    if puzzle_value.startswith(('http://', 'https://')):
        # Custom URL
        clues_data, filename = fetch_clues_from_website(puzzle_value)
    else:
        # Date string - use default cluesbysam.com
        clues_data, filename = fetch_clues_from_website()

        # For date strings, we need to rename the file to match the date
        cache_dir = ".cache"
        expected_filename = os.path.join(cache_dir, f"{puzzle_value}.json")
        if filename != expected_filename:
            os.rename(filename, expected_filename)
            filename = expected_filename

    return clues_data, filename
