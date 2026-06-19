# Pro Football Reference Scraping Fix

## Problem
The `scrape_pfr_game_log()` function was not working because **Pro Football Reference (PFR) is protected by Cloudflare**, which blocks standard HTTP requests with a "Just a moment..." challenge page.

### Root Cause
- When `requests.get()` is called on PFR game log URLs, the server returns HTTP 403
- The response contains a Cloudflare JavaScript challenge that requires browser execution
- Standard HTTP libraries (`requests`, `cloudscraper`, `curl_cffi`) cannot bypass this protection
- The challenge requires actual JavaScript execution to pass, which only a headless browser can handle

## Solution Implemented

### 1. **Cloudflare Detection**
Added `_is_cloudflare_block()` helper function to detect when a response is a Cloudflare challenge page:
```python
def _is_cloudflare_block(resp_text: str) -> bool:
    return "just a moment" in resp_text.lower() or "cloudflare" in resp_text.lower()
```

### 2. **Playwright Fallback**
Implemented `_scrape_pfr_with_playwright()` that uses a headless Chromium browser to pass Cloudflare challenges:
- Launches a headless browser programmatically
- Navigates to the PFR game log URL
- Waits for network to be idle (ensuring JavaScript executes)
- Extracts the HTML after Cloudflare challenge is satisfied

### 3. **Enhanced `scrape_pfr_game_log()` Function**
Updated the main scraping function to:
- First attempt standard `requests` (fast path)
- Detect if response is Cloudflare block (403 status or "Just a moment" text)
- Fall back to Playwright if available
- Return None with clear warning if Playwright is not installed
- Improved error handling and logging at each step

### 4. **Fixed Table Parsing**
Fixed issues in the existing table parsing logic:
- Changed URL construction to include trailing slash (required by PFR)
- Improved table selector fallback order
- Added explicit check for `thead` and `tbody` elements
- Better error messages to identify exactly what went wrong

## Installation

### Option 1: Install Playwright (Recommended for PFR Scraping)
```bash
pip install playwright>=1.40.0
# Then install browser binaries
playwright install chromium
```

### Option 2: Update Requirements
```bash
pip install -r requirements.txt
```

This will install Playwright and other scraping dependencies.

## Required Dependencies Added to requirements.txt
- `beautifulsoup4>=4.12.0` - HTML parsing
- `playwright>=1.40.0` - Cloudflare bypass via headless browser
- `cloudscraper>=1.2.71` - Fallback generic Cloudflare bypass

## Behavior

### When Playwright is Installed
- First request is fast (standard HTTP)
- If Cloudflare blocks, automatically falls back to Playwright
- Playwright handles JavaScript and passes the challenge
- Returns parsed game log data

### When Playwright is NOT Installed
- Standard request works for some sources
- When PFR blocks with Cloudflare, function logs warning and returns None
- Graceful degradation - other scrapers (ESPN, NFL.com, Sleeper) continue to work

## Testing

To test the fix:
```python
from rag.web_scraper import scrape_pfr_game_log

result = scrape_pfr_game_log(
    player_name="Justin Jefferson",
    pfr_player_id="jeffjus01",
    season=2023
)

if result:
    print(f"Success! Got {result.char_count} characters of game log data")
else:
    print("Failed to scrape (check logs for details)")
```

## Notes
- Playwright with headless browser is slower than standard HTTP requests (typically 10-30 seconds per page)
- PFR rate-limiting protection is respected (2x delay after requests)
- The function gracefully degrades if Playwright is not available
- All other scrapers (ESPN, NFL.com, Sleeper API) continue to work normally
