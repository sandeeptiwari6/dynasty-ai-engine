import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from data.schema.database import SourceType
from utils.rag_helpers import init_rag_schema, get_unembedded_docs, get_stale_docs, doc_exists, _clean_text

# Optional: For bypassing Cloudflare protection on PFR
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Polite delay between requests — don't hammer free sources
REQUEST_DELAY_SECONDS = 1.5

# Minimum usable content length — shorter than this is likely a paywall / error page
MIN_CONTENT_CHARS = 200

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class RawDocument:
    """
    Intermediate representation of scraped content before DB insertion.
    All fields map directly to ScoutingDocument columns.
    """
    url: str
    source_type: str
    raw_text: str
    title: str = ""
    player_name: str = ""
    player_id: str = ""
    season: Optional[int] = None
    team: str = ""
    scraped_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_usable(self) -> bool:
        return len(self.raw_text.strip()) >= MIN_CONTENT_CHARS

    @property
    def char_count(self) -> int:
        return len(self.raw_text)

def scrape_nfl_player_bio(player_name: str, nfl_player_slug: str) -> Optional[RawDocument]:
    """
    Scrape the player bio page from NFL.com.
    URL pattern: https://www.nfl.com/players/{first-last}/

    Args:
        player_name: Display name (e.g. "Justin Jefferson")
        nfl_player_slug: URL slug (e.g. "justin-jefferson")

    Returns a RawDocument with career overview text, or None on failure.
    """
    url = f"https://www.nfl.com/players/{nfl_player_slug}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # NFL.com bio structure: player-detail__header + nfl-c-player-info blocks
        sections = []

        # Player header (name, position, team, status)
        header = soup.select_one(".player-detail__header")
        if header:
            sections.append(header.get_text(separator=" ", strip=True))

        # Bio / career overview text block
        for selector in [
            ".nfl-c-player-bio__text",
            ".player-bio",
            "[class*='bio']",
            ".player-detail__bio",
        ]:
            bio = soup.select_one(selector)
            if bio:
                sections.append(bio.get_text(separator="\n", strip=True))
                break

        # Stats summary table (convert to readable text)
        stats_table = soup.select_one(".nfl-o-roster__table, .player-stats")
        if stats_table:
            rows = stats_table.find_all("tr")
            for row in rows[:10]:  # cap at 10 rows
                cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
                if cells:
                    sections.append(" | ".join(cells))

        full_text = "\n\n".join(s for s in sections if s)
        title = soup.title.string if soup.title else player_name

        time.sleep(REQUEST_DELAY_SECONDS)

        logger.info(f"Successfully scraped NFL.com bio for {player_name}")
        return RawDocument(
            url=url,
            source_type=SourceType.NFL_COM_BIO,
            raw_text=full_text or f"NFL.com bio for {player_name} — no text extracted.",
            title=title,
            player_name=player_name,
        )

    except requests.RequestException as e:
        logger.warning(f"NFL.com bio scrape failed for {player_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# 2. Sleeper news feed
# ---------------------------------------------------------------------------

def scrape_sleeper_news(player_sleeper_id: str, player_name: str) -> list[RawDocument]:
    """
    Pull recent news items from the Sleeper API news endpoint.
    Returns one RawDocument per news item (not per player) — each item is
    short but self-contained and should be a separate chunk.

    Sleeper news items are the highest-signal short-form source:
    they cover injuries, depth chart changes, FA signings, and practice notes
    in 1-3 sentences each, sourced from beat reporters.

    No API key required.
    """
    url = f"https://api.sleeper.app/v1/players/nfl/{player_sleeper_id}/news"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        items = resp.json()
    except Exception as e:
        logger.warning(f"Sleeper news failed for {player_name} ({player_sleeper_id}): {e}")
        return []

    docs = []
    for item in items[:20]:  # most recent 20 items
        # Sleeper item structure: {title, analysis, source, published}
        title = item.get("title", "")
        analysis = item.get("analysis", "")
        source = item.get("source", "")
        published = item.get("published", 0)
        pub_year = datetime.fromtimestamp(published / 1000).year if published else None

        body = f"{title}\n\n{analysis}"
        if source:
            body += f"\n\nSource: {source}"

        # Determine source type from content signals
        source_type = _classify_sleeper_item(title, analysis)

        # Use published timestamp as the item URL surrogate (no permanent URL in API)
        item_url = f"sleeper://player/{player_sleeper_id}/news/{published}"

        docs.append(RawDocument(
            url=item_url,
            source_type=source_type,
            raw_text=body,
            title=title,
            player_name=player_name,
            player_id=player_sleeper_id,
            season=pub_year,
        ))
        logger.info(f"Scraped Sleeper news item for {player_name}: {title} ({source_type})")
    time.sleep(REQUEST_DELAY_SECONDS * 0.3)  # Sleeper is our own API, be lighter
    return docs


def _classify_sleeper_item(title: str, body: str) -> str:
    """Classify a Sleeper news item into a SourceType based on keywords."""
    combined = (title + " " + body).lower()
    if any(w in combined for w in ["injured", "injury", "questionable", "out", "ir", "placed"]):
        return SourceType.INJURY_REPORT
    if any(w in combined for w in ["signed", "trade", "released", "waived", "cut", "deal"]):
        return SourceType.TRANSACTION_NOTE
    if any(w in combined for w in ["depth chart", "starter", "backup", "role", "snap"]):
        return SourceType.DEPTH_CHART
    return SourceType.BEAT_REPORTER


# ---------------------------------------------------------------------------
# 3. ESPN / NFL draft profiles
# ---------------------------------------------------------------------------

def scrape_espn_draft_profile(player_name: str, espn_player_id: str) -> Optional[RawDocument]:
    """
    Scrape an ESPN draft prospect profile page.
    URL pattern: https://www.espn.com/nfl/player/_/id/{id}/{slug}

    ESPN draft profiles contain:
      - Position ranking, overall ranking
      - Scouting report text (strengths/weaknesses)
      - Measurables (height, weight, 40)
      - College stats summary

    This is the richest open-source text for prospect evaluation.
    """
    slug = player_name.lower().replace(" ", "-").replace("'", "").replace(".", "")
    url = f"https://www.espn.com/nfl/player/_/id/{espn_player_id}/{slug}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        sections = []

        # Player header block
        for sel in [".PlayerHeader", ".player-header", "[class*='PlayerHeader']"]:
            header = soup.select_one(sel)
            if header:
                sections.append(header.get_text(separator=" ", strip=True))
                break

        # Bio / scouting text — ESPN wraps this in article or section tags
        for sel in [
            ".article-body",
            "[class*='scouting']",
            "[class*='bio']",
            ".player__bio",
            "article",
        ]:
            content = soup.select_one(sel)
            if content and len(content.get_text(strip=True)) > MIN_CONTENT_CHARS:
                # Strip nav/script tags before extracting text
                for tag in content.find_all(["script", "style", "nav"]):
                    tag.decompose()
                sections.append(content.get_text(separator="\n", strip=True))
                break

        # Stats table
        stats = soup.select_one(".Table, [class*='stats']")
        if stats:
            rows = stats.find_all("tr")[:8]
            for row in rows:
                cells = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
                if cells:
                    sections.append(" | ".join(cells))

        full_text = "\n\n".join(s for s in sections if s.strip())
        title = soup.title.string if soup.title else f"{player_name} Draft Profile"

        time.sleep(REQUEST_DELAY_SECONDS)

        logger.info(f"Successfully scraped ESPN draft profile for {player_name}")
        return RawDocument(
            url=url,
            source_type=SourceType.DRAFT_PROFILE,
            raw_text=full_text or f"Draft profile for {player_name}.",
            title=title,
            player_name=player_name,
        )

    except requests.RequestException as e:
        logger.warning(f"ESPN draft profile scrape failed for {player_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# 4. Pro Football Reference game log → narrative recap
# ---------------------------------------------------------------------------

def _is_cloudflare_block(resp_text: str) -> bool:
    """Detect if response is a Cloudflare challenge page."""
    return "just a moment" in resp_text.lower() or "cloudflare" in resp_text.lower()


def _scrape_pfr_with_playwright(
    player_name: str,
    pfr_player_id: str,
    season: int,
) -> Optional[str]:
    """
    Scrape PFR using Playwright to handle Cloudflare JavaScript challenge.
    Returns raw HTML on success, None on failure.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return None
    
    first_letter = pfr_player_id[0].upper()
    url = f"https://www.pro-football-reference.com/players/{first_letter}/{pfr_player_id}/gamelog/{season}/"
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Set timeout and navigate
            page.set_default_timeout(30000)  # 30 seconds
            page.goto(url, wait_until="networkidle")
            
            # Get page content
            html = page.content()
            browser.close()
            logger.info(f"Successfully scraped PFR game log for {player_name} {season} with Playwright")
            return html
    except Exception as e:
        logger.warning(f"Playwright scrape failed for {player_name} {season}: {e}")
        return None


def scrape_pfr_game_log(
    player_name: str,
    pfr_player_id: str,
    season: int,
) -> Optional[RawDocument]:
    """
    Pull season game log from Pro Football Reference and convert to
    narrative text that the LLM can reason over.

    PFR URL: https://www.pro-football-reference.com/players/{X}/{pfr_id}/gamelog/{season}/

    We convert the structured table into readable sentences because
    raw HTML tables chunk poorly in RAG — the LLM does better with prose.

    NOTE: PFR is protected by Cloudflare. This function attempts standard
    requests first, then falls back to Playwright (if available) for
    JavaScript-based challenges.

    Example output:
      "In 2023, Justin Jefferson played 17 games. Week 1 vs TB: 9 targets,
       7 receptions, 150 yards, 1 TD. Week 2 at PHI: 6 targets, 4 rec, 57 yds..."
    """
    first_letter = pfr_player_id[0].upper()
    url = f"https://www.pro-football-reference.com/players/{first_letter}/{pfr_player_id}/gamelog/{season}/"

    html = None

    # Try standard requests first (faster)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        
        # Check for Cloudflare block
        if resp.status_code == 403 or _is_cloudflare_block(resp.text):
            logger.info(f"PFR: Cloudflare challenge detected for {player_name}, attempting Playwright...")
            
            # Try Playwright as fallback
            if PLAYWRIGHT_AVAILABLE:
                html = _scrape_pfr_with_playwright(player_name, pfr_player_id, season)
            else:
                logger.warning(
                    f"PFR game log blocked by Cloudflare for {player_name} {season}. "
                    "Install playwright for Cloudflare bypass: pip install playwright"
                )
                return None
        else:
            html = resp.text
            
    except requests.RequestException as e:
        logger.warning(f"PFR request failed for {player_name} {season}: {e}")
        return None
    
    if not html:
        return None

    # Parse HTML
    try:
        soup = BeautifulSoup(html, "lxml")

        # Game log table — try multiple selectors
        table = None
        for selector in ["#stats", "#receiving", "#rushing", "#passing", "table[id*='gamelog']"]:
            table = soup.select_one(selector)
            if table:
                break
        
        # Fallback: use first table if no ID-based selector worked
        if not table:
            tables = soup.find_all("table")
            if tables:
                table = tables[0]

        if not table:
            logger.warning(f"PFR: no game log table found for {player_name} {season}")
            return None

        # Parse headers
        headers_row = table.select_one("thead tr")
        if not headers_row:
            logger.warning(f"PFR: no table headers found for {player_name} {season}")
            return None
        col_headers = [th.get_text(strip=True) for th in headers_row.find_all(["th", "td"])]

        # Parse rows
        game_lines = []
        tbody = table.find("tbody")
        if not tbody:
            logger.warning(f"PFR: no table body found for {player_name} {season}")
            return None
            
        for row in tbody.find_all("tr"):
            if "thead" in row.get("class", []) or "partial_table" in row.get("class", []):
                continue
            cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
            if not cells or not cells[0]:
                continue

            # Map cells to headers and build a readable sentence
            row_data = dict(zip(col_headers, cells))
            line = _pfr_row_to_prose(row_data, player_name)
            if line:
                game_lines.append(line)

        if not game_lines:
            logger.warning(f"PFR: no game rows found for {player_name} {season}")
            return None

        full_text = (
            f"{player_name} — {season} NFL Season Game Log\n\n"
            + "\n".join(game_lines)
        )

        time.sleep(REQUEST_DELAY_SECONDS * 2)  # PFR is rate-sensitive

        return RawDocument(
            url=url,
            source_type=SourceType.GAME_RECAP,
            raw_text=full_text,
            title=f"{player_name} {season} Game Log",
            player_name=player_name,
            season=season,
        )

    except Exception as e:
        logger.error(f"PFR parsing failed for {player_name} {season}: {e}")
        return None


def _pfr_row_to_prose(row: dict, player_name: str) -> str:
    """
    Convert a single PFR game log row dict into a readable sentence.
    """
    week = row.get("Week", row.get("Wk", "?"))
    opp = row.get("Opp", "")
    result = row.get("Result", "")

    # Detect position from available columns
    if "Yds" in row and "Rec" in row:  # receiver
        tgt = row.get("Tgt", "?")
        rec = row.get("Rec", "?")
        yds = row.get("Yds", "?")
        td = row.get("TD", "0")
        return f"Week {week} vs {opp} ({result}): {tgt} targets, {rec} rec, {yds} yds, {td} TD"
    elif "Att" in row and "Yds" in row and "Car" not in row:  # passer
        comp = row.get("Cmp", "?")
        att = row.get("Att", "?")
        yds = row.get("Yds", "?")
        td = row.get("TD", "0")
        ints = row.get("Int", "0")
        return f"Week {week} vs {opp} ({result}): {comp}/{att}, {yds} yds, {td} TD, {ints} INT"
    elif "Car" in row:  # rusher
        car = row.get("Car", "?")
        yds = row.get("Yds", "?")
        td = row.get("TD", "0")
        return f"Week {week} vs {opp} ({result}): {car} carries, {yds} yds, {td} TD"
    return ""


# ---------------------------------------------------------------------------
# 5. Generic URL fetcher — for beat reporter articles you supply manually
# ---------------------------------------------------------------------------

def scrape_generic_article(
    url: str,
    player_name: str,
    player_id: str = "",
    season: Optional[int] = None,
    source_type: str = SourceType.BEAT_REPORTER,
) -> Optional[RawDocument]:
    """
    Generic scraper for any article URL.
    Extracts main content using common article body selectors.

    Supports: The Athletic, ESPN, NFL.com articles, local beat reporter sites.
    Does NOT support: paywalled content (will return short text → marked unusable).
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise elements before extracting text
        for tag in soup.find_all(["script", "style", "nav", "footer",
                                   "header", "aside", "iframe", "noscript"]):
            tag.decompose()

        # Try known article content selectors in priority order
        content_text = ""
        for selector in [
            "article",
            "[class*='article-body']",
            "[class*='story-body']",
            "[class*='article__body']",
            "[class*='content-body']",
            "main",
            ".post-content",
        ]:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator="\n", strip=True)
                if len(text) > MIN_CONTENT_CHARS:
                    content_text = text
                    break

        # Fallback: all paragraph text
        if not content_text:
            paragraphs = soup.find_all("p")
            content_text = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # Clean up text
        content_text = _clean_text(content_text)

        time.sleep(REQUEST_DELAY_SECONDS)

        return RawDocument(
            url=url,
            source_type=source_type,
            raw_text=content_text,
            title=title,
            player_name=player_name,
            player_id=player_id,
            season=season,
        )

    except requests.RequestException as e:
        logger.warning(f"Generic scrape failed for {url}: {e}")
        return None
