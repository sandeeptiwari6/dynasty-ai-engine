# Setting Up Playwright for PFR Scraping

## Quick Setup

Playwright is now an optional dependency for bypassing Cloudflare protection on Pro Football Reference. Here's how to install it:

### Step 1: Install Playwright Package
```bash
pip install playwright>=1.40.0
```

### Step 2: Install Browser Binaries
```bash
playwright install chromium
```

Or install all browsers:
```bash
playwright install
```

## Verification

To verify Playwright is working:
```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://example.com")
    print(page.title())
    browser.close()
```

## What Gets Installed

Running `playwright install` downloads:
- **Chromium** (~150 MB) - Recommended, fastest
- **Firefox** (~150 MB) - Alternative
- **Webkit** (~180 MB) - Alternative

For PFR, Chromium is used by default and is sufficient.

## Troubleshooting

### Issue: "No module named 'playwright'"
Solution: Run `pip install playwright>=1.40.0`

### Issue: "Browser not found"
Solution: Run `playwright install chromium`

### Issue: Installation fails on macOS
Solution: You may need Xcode command line tools:
```bash
xcode-select --install
```

Then retry `playwright install chromium`

### Issue: Playwright times out on PFR
- This is expected if PFR's JavaScript loads slowly
- The timeout is set to 30 seconds in the code
- Check your internet connection

## Optional: Using with Other Data Sources

The code gracefully handles Playwright being unavailable:
- If Playwright is NOT installed, the code logs a warning but doesn't crash
- Other data sources (ESPN, NFL.com, Sleeper API) continue to work
- Only PFR scraping requires Playwright

## Alternative: Without Playwright

If you don't want to install Playwright:
1. Comment out Playwright imports in `rag/web_scraper.py`
2. PFR scraping will be skipped (logged as warning)
3. All other data sources continue to work normally

## Performance Notes

- Standard HTTP requests: ~1-3 seconds per page
- Playwright requests: ~10-30 seconds per page (includes browser startup)
- Browser is launched and closed for each request (intentional for safety)

## Next Steps

After setup, the PFR scraper will automatically use Playwright when needed:
```python
from rag.web_scraper import scrape_pfr_game_log

# This will now work even with Cloudflare protection
result = scrape_pfr_game_log(
    player_name="Justin Jefferson",
    pfr_player_id="jeffjus01", 
    season=2023
)
```
