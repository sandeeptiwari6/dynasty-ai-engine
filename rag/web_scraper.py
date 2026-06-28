import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from data.schema.database import SourceType
from utils.rag_helpers import init_rag_schema, get_unembedded_docs, get_stale_docs, doc_exists, _clean_text

# Optional: Playwright for PFR fallback (bypasses Cloudflare)
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)

REQUEST_DELAY_SECONDS = 1.5
MIN_CONTENT_CHARS = 200

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

# Sleeper all-players data (14 MB, cached 1 hr)
_SLEEPER_CACHE: dict = {}
_SLEEPER_CACHE_TS: float = 0.0
_SLEEPER_CACHE_TTL = 3600

# nfl_data_py weekly data per season (cached indefinitely in process lifetime)
_NFL_WEEKLY_CACHE: dict = {}


@dataclass
class RawDocument:
    """Intermediate representation of scraped content before DB insertion."""
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


# ---------------------------------------------------------------------------
# 1. NFL.com player bio
# ---------------------------------------------------------------------------

def scrape_nfl_player_bio(player_name: str, nfl_player_slug: str) -> Optional[RawDocument]:
    """
    Scrape the player bio page from NFL.com.
    URL pattern: https://www.nfl.com/players/{first-last}/
    """
    url = f"https://www.nfl.com/players/{nfl_player_slug}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        sections = []

        # Page-level description (gives search-engine summary of the player page)
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc and meta_desc.get("content"):
            sections.append(meta_desc["content"])

        # Player header — actual NFL.com CSS classes (as of 2024-2025 site)
        header = soup.select_one(".nfl-c-player-header")
        if header:
            header_parts = []
            for sel, label in [
                (".nfl-c-player-header__title", "Player"),
                (".nfl-c-player-header__position", "Position"),
                (".nfl-c-player-header__team", "Team"),
                (".nfl-c-player-header__roster-status", "Roster Status"),
            ]:
                el = header.select_one(sel)
                if el:
                    val = el.get_text(strip=True)
                    if val:
                        header_parts.append(f"{label}: {val}")
            if header_parts:
                sections.append("\n".join(header_parts))

        # Player info — physical + career key-value pairs
        info = soup.select_one(".nfl-c-player-info")
        if info:
            keys = info.select(".nfl-c-player-info__key")
            vals = info.select(".nfl-c-player-info__value")
            kv = []
            for k, v in zip(keys, vals):
                k_text = k.get_text(strip=True)
                v_text = v.get_text(strip=True)
                if k_text and v_text:
                    kv.append(f"{k_text}: {v_text}")
            if kv:
                sections.append("\n".join(kv))

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
# 2. Sleeper player status + ESPN news/rotowire
#    The old /v1/players/nfl/{id}/news endpoint was deprecated (returns 404).
#    We now pull:
#      (a) Current injury/depth-chart status from Sleeper's all-players endpoint
#      (b) ESPN rotowire update + recent articles using the ESPN ID stored in
#          Sleeper player data
# ---------------------------------------------------------------------------

def _load_sleeper_cache() -> None:
    """Fetch the Sleeper all-players payload and store it in module cache."""
    global _SLEEPER_CACHE, _SLEEPER_CACHE_TS
    try:
        resp = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=60)
        if resp.status_code == 200:
            _SLEEPER_CACHE = resp.json()
            _SLEEPER_CACHE_TS = time.time()
            logger.info(f"Loaded Sleeper players cache: {len(_SLEEPER_CACHE)} players")
        else:
            logger.warning(f"Sleeper all-players returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to load Sleeper players cache: {e}")


def _get_sleeper_player(sleeper_id: str) -> dict:
    """Return one player's data dict from the Sleeper cache (lazy-load)."""
    global _SLEEPER_CACHE, _SLEEPER_CACHE_TS
    if not _SLEEPER_CACHE or (time.time() - _SLEEPER_CACHE_TS) > _SLEEPER_CACHE_TTL:
        _load_sleeper_cache()
    return _SLEEPER_CACHE.get(str(sleeper_id), {})


def scrape_sleeper_news(player_sleeper_id: str, player_name: str) -> list[RawDocument]:
    """
    Return player status + ESPN news documents for a player.

    Strategy (Sleeper /news endpoint was removed):
      1. Pull the player's current status fields from Sleeper all-players data
         (injury status, depth chart, team, etc.) → one RawDocument.
      2. Use the ESPN athlete ID stored in Sleeper data to pull ESPN rotowire
         update and recent news articles via ESPN's JSON API → one doc per item.
    """
    docs: list[RawDocument] = []

    player_data = _get_sleeper_player(player_sleeper_id)

    # --- Sleeper status document ---
    if player_data:
        status_lines = [f"{player_name} — Current Player Status"]
        for label, key in [
            ("Team", "team"),
            ("Position", "position"),
            ("Active Status", "status"),
            ("Age", "age"),
            ("Height (inches)", "height"),
            ("Weight (lbs)", "weight"),
            ("College", "college"),
            ("Years of Experience", "years_exp"),
            ("Jersey Number", "number"),
            ("Depth Chart Position", "depth_chart_position"),
            ("Depth Chart Order", "depth_chart_order"),
            ("Injury Status", "injury_status"),
            ("Injury Body Part", "injury_body_part"),
            ("Injury Notes", "injury_notes"),
        ]:
            val = player_data.get(key)
            if val not in (None, ""):
                status_lines.append(f"{label}: {val}")

        status_text = "\n".join(status_lines)

        injury_status = player_data.get("injury_status")
        depth_order = player_data.get("depth_chart_order")
        if injury_status:
            src_type = SourceType.INJURY_REPORT
        elif depth_order is not None:
            src_type = SourceType.DEPTH_CHART
        else:
            src_type = SourceType.BEAT_REPORTER

        docs.append(RawDocument(
            url=f"sleeper://player/{player_sleeper_id}/status",
            source_type=src_type,
            raw_text=status_text,
            title=f"{player_name} Player Status",
            player_name=player_name,
            player_id=str(player_sleeper_id),
        ))

        # --- ESPN news + rotowire using ESPN ID from Sleeper data ---
        espn_id = player_data.get("espn_id")
        if espn_id:
            espn_docs = _scrape_espn_news_and_rotowire(int(espn_id), player_name, str(player_sleeper_id))
            docs.extend(espn_docs)

    if not docs:
        logger.warning(f"No Sleeper/ESPN data for {player_name} (sleeper_id={player_sleeper_id})")

    time.sleep(REQUEST_DELAY_SECONDS * 0.3)
    return docs


def _scrape_espn_news_and_rotowire(
    espn_id: int,
    player_name: str,
    player_id: str,
) -> list[RawDocument]:
    """
    Pull rotowire update and recent news from ESPN's player overview JSON API.
    Returns up to 11 RawDocuments (1 rotowire + 10 articles).
    """
    url = f"https://site.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{espn_id}/overview"
    docs: list[RawDocument] = []

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Rotowire (latest injury/transaction note)
        rotowire = data.get("rotowire") or {}
        if rotowire.get("story"):
            headline = rotowire.get("headline", "")
            story = rotowire.get("story", "")
            published = rotowire.get("published", "")
            body = f"{headline}\n\n{story}"
            if published:
                body += f"\n\nPublished: {published}"

            docs.append(RawDocument(
                url=f"espn://athletes/{espn_id}/rotowire",
                source_type=_classify_sleeper_item(headline, story),
                raw_text=body,
                title=f"{player_name} — ESPN Rotowire",
                player_name=player_name,
                player_id=player_id,
            ))

        # Recent news articles
        for item in (data.get("news") or [])[:10]:
            headline = item.get("headline", "")
            description = item.get("description", "")
            published = item.get("lastModified", "")
            item_url = (
                item.get("links", {}).get("web", {}).get("href")
                or f"espn://athletes/{espn_id}/news/{item.get('id', '')}"
            )

            body = f"{headline}\n\n{description}"
            if published:
                body += f"\n\nPublished: {published}"

            docs.append(RawDocument(
                url=item_url,
                source_type=SourceType.BEAT_REPORTER,
                raw_text=body,
                title=headline or f"{player_name} News",
                player_name=player_name,
                player_id=player_id,
            ))

        logger.info(f"ESPN overview: {len(docs)} docs for {player_name} (ESPN {espn_id})")

    except Exception as e:
        logger.warning(f"ESPN news/rotowire failed for {player_name} (ESPN {espn_id}): {e}")

    return docs


def _classify_sleeper_item(title: str, body: str) -> str:
    """Classify a news item into a SourceType based on keywords."""
    combined = (title + " " + body).lower()
    if any(w in combined for w in ["injured", "injury", "questionable", "out", "ir", "placed"]):
        return SourceType.INJURY_REPORT
    if any(w in combined for w in ["signed", "trade", "released", "waived", "cut", "deal"]):
        return SourceType.TRANSACTION_NOTE
    if any(w in combined for w in ["depth chart", "starter", "backup", "role", "snap"]):
        return SourceType.DEPTH_CHART
    return SourceType.BEAT_REPORTER


# ---------------------------------------------------------------------------
# 3. Game log — nfl_data_py (primary) + PFR/Playwright (fallback)
#    PFR is now protected by Cloudflare Turnstile, which blocks all automated
#    HTTP requests and headless browsers.  nfl_data_py pulls the same underlying
#    nflverse data without any scraping, so it is the correct primary source.
# ---------------------------------------------------------------------------

def _get_nfl_weekly_data(season: int):
    """Lazy-load nfl_data_py weekly stats for a season (cached per process)."""
    global _NFL_WEEKLY_CACHE
    if season not in _NFL_WEEKLY_CACHE:
        try:
            import nfl_data_py as nfl  # imported here to keep it optional
            cols = [
                "player_id", "player_display_name", "position",
                "season", "week", "recent_team", "opponent_team",
                "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
                "carries", "rushing_yards", "rushing_tds",
                "receptions", "targets", "receiving_yards", "receiving_tds",
                "fantasy_points_ppr",
            ]
            df = nfl.import_weekly_data([season], columns=cols)
            _NFL_WEEKLY_CACHE[season] = df
            logger.info(f"nfl_data_py: loaded {len(df)} rows for {season}")
        except Exception as e:
            logger.warning(f"nfl_data_py import failed for {season}: {e}")
            _NFL_WEEKLY_CACHE[season] = None
    return _NFL_WEEKLY_CACHE[season]


def scrape_pfr_game_log(
    player_name: str,
    pfr_player_id: str,
    season: int,
    gsis_id: str = "",
) -> Optional[RawDocument]:
    """
    Return a season game log as narrative prose.

    Primary source: nfl_data_py (nflverse data, no scraping).
    Fallback: PFR via Playwright headless browser (only if nfl_data_py fails and
              PLAYWRIGHT_AVAILABLE is True).

    Args:
        player_name: Display name (e.g. "Justin Jefferson")
        pfr_player_id: PFR ID (e.g. "JeffJu00") — used only by the Playwright fallback
        season: NFL season year
        gsis_id: GSIS/nflverse player_id (e.g. "00-0036322") — preferred for exact lookup
    """
    doc = _game_log_from_nfl_data_py(player_name, season, gsis_id)
    if doc:
        return doc

    logger.info(f"nfl_data_py miss for {player_name} {season} — trying PFR/Playwright fallback")
    return _scrape_pfr_via_playwright(player_name, pfr_player_id, season)


def _game_log_from_nfl_data_py(
    player_name: str,
    season: int,
    gsis_id: str = "",
) -> Optional[RawDocument]:
    """Look up one player's weekly game log from the nfl_data_py cache."""
    df = _get_nfl_weekly_data(season)
    if df is None or len(df) == 0:
        return None

    # Prefer exact GSIS ID match; fall back to display name
    if gsis_id:
        player_df = df[df["player_id"] == gsis_id]
    else:
        player_df = df[df["player_display_name"].str.lower() == player_name.lower()]

    # Last-name fuzzy fallback (handles cases like "Patrick Mahomes II")
    if player_df.empty:
        last = player_name.split()[-1].lower() if player_name else ""
        if last:
            candidates = df[df["player_display_name"].str.lower().str.contains(last, regex=False)]
            unique_names = candidates["player_display_name"].unique()
            if len(unique_names) == 1:
                player_df = candidates
            else:
                logger.debug(f"nfl_data_py: ambiguous/no match for '{player_name}' in {season}")
                return None

    if player_df.empty:
        return None

    position = player_df["position"].iloc[0] if "position" in player_df.columns else ""
    game_lines = []

    for _, row in player_df.sort_values("week").iterrows():
        line = _nfl_row_to_prose(row, position)
        if line:
            game_lines.append(line)

    if not game_lines:
        return None

    display_name = player_df["player_display_name"].iloc[0]
    full_text = f"{display_name} — {season} NFL Season Game Log\n\n" + "\n".join(game_lines)
    url = f"nflverse://players/{gsis_id or player_name.replace(' ', '_')}/gamelog/{season}"

    logger.info(f"nfl_data_py game log: {player_name} {season} ({len(game_lines)} games)")
    return RawDocument(
        url=url,
        source_type=SourceType.GAME_RECAP,
        raw_text=full_text,
        title=f"{display_name} {season} Game Log",
        player_name=player_name,
        season=season,
    )


def _nfl_row_to_prose(row, position: str) -> str:
    """Convert one nfl_data_py weekly row to a readable sentence."""
    week = int(row.get("week") or 0)
    opp = str(row.get("opponent_team") or "")
    fpts = float(row.get("fantasy_points_ppr") or 0)

    att = int(row.get("attempts") or 0)
    carries = int(row.get("carries") or 0)
    tgt = int(row.get("targets") or 0)

    if position == "QB" or att > 0:
        comp = int(row.get("completions") or 0)
        yds = int(row.get("passing_yards") or 0)
        td = int(row.get("passing_tds") or 0)
        ints = int(row.get("interceptions") or 0)
        if att == 0:
            return ""
        return f"Week {week} vs {opp}: {comp}/{att}, {yds} yds, {td} TD, {ints} INT ({fpts:.1f} PPR pts)"

    if position == "RB" or (carries > 0 and tgt == 0):
        yds = int(row.get("rushing_yards") or 0)
        td = int(row.get("rushing_tds") or 0)
        rec = int(row.get("receptions") or 0)
        rec_yds = int(row.get("receiving_yards") or 0)
        if carries == 0 and rec == 0:
            return ""
        line = f"Week {week} vs {opp}: {carries} car/{yds} rush yds/{td} TD"
        if rec:
            line += f", {rec} rec/{rec_yds} yds"
        return line + f" ({fpts:.1f} PPR pts)"

    # WR / TE / flex receiver
    if tgt > 0 or position in ("WR", "TE"):
        rec = int(row.get("receptions") or 0)
        yds = int(row.get("receiving_yards") or 0)
        td = int(row.get("receiving_tds") or 0)
        if carries > 0:
            rush_yds = int(row.get("rushing_yards") or 0)
            return (f"Week {week} vs {opp}: {rec}/{tgt} rec, {yds} yds, {td} TD, "
                    f"{carries} car/{rush_yds} rush yds ({fpts:.1f} PPR pts)")
        return f"Week {week} vs {opp}: {rec}/{tgt} rec, {yds} yds, {td} TD ({fpts:.1f} PPR pts)"

    return ""


def _is_cloudflare_block(resp_text: str) -> bool:
    return "just a moment" in resp_text.lower() or "cloudflare" in resp_text.lower()


def _scrape_pfr_via_playwright(
    player_name: str,
    pfr_player_id: str,
    season: int,
) -> Optional[RawDocument]:
    """
    Playwright-based PFR fallback.  Uses stealth settings to reduce bot detection.
    Note: Cloudflare Turnstile on PFR may still block headless browsers.
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.warning(
            f"PFR fallback unavailable for {player_name} {season}: playwright not installed."
        )
        return None

    first_letter = pfr_player_id[0].upper() if pfr_player_id else ""
    if not first_letter:
        return None
    url = f"https://www.pro-football-reference.com/players/{first_letter}/{pfr_player_id}/gamelog/{season}/"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            context = browser.new_context(
                user_agent=HEADERS["User-Agent"],
                extra_http_headers={"Accept-Language": HEADERS["Accept-Language"]},
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()
            page.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
                "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});"
            )

            # Use domcontentloaded (not networkidle) — Cloudflare JS may keep network busy
            page.goto(url, wait_until="domcontentloaded", timeout=40000)
            time.sleep(5)  # wait for Cloudflare challenge to auto-resolve
            html = page.content()
            browser.close()

        if _is_cloudflare_block(html):
            logger.warning(f"PFR Cloudflare block persists for {player_name} {season}")
            return None

        # Parse HTML exactly as before
        soup = BeautifulSoup(html, "lxml")
        table = None
        for selector in ["#stats", "#receiving", "#rushing", "#passing", "table[id*='gamelog']"]:
            table = soup.select_one(selector)
            if table:
                break
        if not table:
            tables = soup.find_all("table")
            table = tables[0] if tables else None
        if not table:
            return None

        headers_row = table.select_one("thead tr")
        if not headers_row:
            return None
        col_headers = [th.get_text(strip=True) for th in headers_row.find_all(["th", "td"])]

        game_lines = []
        tbody = table.find("tbody")
        if not tbody:
            return None
        for row in tbody.find_all("tr"):
            if "thead" in row.get("class", []) or "partial_table" in row.get("class", []):
                continue
            cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
            if not cells or not cells[0]:
                continue
            row_data = dict(zip(col_headers, cells))
            line = _pfr_row_to_prose(row_data, player_name)
            if line:
                game_lines.append(line)

        if not game_lines:
            return None

        time.sleep(REQUEST_DELAY_SECONDS * 2)
        full_text = (
            f"{player_name} — {season} NFL Season Game Log (via PFR)\n\n"
            + "\n".join(game_lines)
        )
        return RawDocument(
            url=url,
            source_type=SourceType.GAME_RECAP,
            raw_text=full_text,
            title=f"{player_name} {season} Game Log",
            player_name=player_name,
            season=season,
        )

    except Exception as e:
        logger.warning(f"PFR Playwright fallback failed for {player_name} {season}: {e}")
        return None


def _pfr_row_to_prose(row: dict, player_name: str) -> str:
    """Convert a single PFR game log row dict into a readable sentence."""
    week = row.get("Week", row.get("Wk", "?"))
    opp = row.get("Opp", "")
    result = row.get("Result", "")
    if "Yds" in row and "Rec" in row:
        tgt = row.get("Tgt", "?")
        rec = row.get("Rec", "?")
        yds = row.get("Yds", "?")
        td = row.get("TD", "0")
        return f"Week {week} vs {opp} ({result}): {tgt} targets, {rec} rec, {yds} yds, {td} TD"
    elif "Att" in row and "Yds" in row and "Car" not in row:
        comp = row.get("Cmp", "?")
        att = row.get("Att", "?")
        yds = row.get("Yds", "?")
        td = row.get("TD", "0")
        ints = row.get("Int", "0")
        return f"Week {week} vs {opp} ({result}): {comp}/{att}, {yds} yds, {td} TD, {ints} INT"
    elif "Car" in row:
        car = row.get("Car", "?")
        yds = row.get("Yds", "?")
        td = row.get("TD", "0")
        return f"Week {week} vs {opp} ({result}): {car} carries, {yds} yds, {td} TD"
    return ""


# ---------------------------------------------------------------------------
# 4. ESPN player profile (JSON API)
#    The ESPN HTML pages are behind bot detection (returns 202 + empty body).
#    The ESPN common API returns structured JSON without anti-bot checks.
# ---------------------------------------------------------------------------

def scrape_espn_draft_profile(player_name: str, espn_player_id: str) -> Optional[RawDocument]:
    """
    Fetch ESPN player profile using ESPN's JSON overview API.

    ESPN HTML pages are bot-protected; the JSON API is open.
    Returns bio (awards, team history), rotowire note, and stats summary.
    """
    if not espn_player_id:
        return None

    sections = []

    # Overview API (stats, rotowire, recent news)
    overview_url = f"https://site.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{espn_player_id}/overview"
    # Bio API (awards, team history)
    bio_url = f"https://site.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{espn_player_id}/bio"

    try:
        bio_resp = requests.get(bio_url, headers=HEADERS, timeout=10)
        if bio_resp.status_code == 200:
            bio = bio_resp.json()
            awards = bio.get("awards", [])
            if awards:
                award_strs = [
                    f"{a['name']} ({', '.join(a.get('seasons', []))})"
                    for a in awards
                ]
                sections.append("Awards: " + ", ".join(award_strs))
            team_history = bio.get("teamHistory", [])
            if team_history:
                history_lines = [
                    f"  {t['displayName']} ({t.get('seasons', '')})"
                    for t in team_history
                ]
                sections.append("Team History:\n" + "\n".join(history_lines))

        overview_resp = requests.get(overview_url, headers=HEADERS, timeout=15)
        overview_resp.raise_for_status()
        data = overview_resp.json()

        # Rotowire latest note
        rotowire = data.get("rotowire") or {}
        if rotowire.get("story"):
            sections.append(
                f"Latest Update ({rotowire.get('published', '')}):\n"
                f"{rotowire['headline']}\n{rotowire['story']}"
            )

        # Recent game log summary (last 5 games)
        game_log = data.get("gameLog") or {}
        stats_blocks = game_log.get("statistics") or []
        if stats_blocks:
            stat_block = stats_blocks[0]
            labels = stat_block.get("labels", [])
            events = stat_block.get("events", [])[:5]
            if labels and events:
                gl_lines = [f"Recent games ({stat_block.get('displayName', 'Stats')})"]
                for ev in events:
                    ev_stats = ev.get("stats", [])
                    if ev_stats:
                        gl_lines.append("  " + " | ".join(
                            f"{l}: {s}" for l, s in zip(labels, ev_stats)
                        ))
                sections.append("\n".join(gl_lines))

        full_text = "\n\n".join(s for s in sections if s.strip())

        time.sleep(REQUEST_DELAY_SECONDS)

        logger.info(f"ESPN JSON profile fetched for {player_name} (ID {espn_player_id})")
        return RawDocument(
            url=overview_url,
            source_type=SourceType.DRAFT_PROFILE,
            raw_text=full_text or f"ESPN profile for {player_name}.",
            title=f"{player_name} ESPN Profile",
            player_name=player_name,
        )

    except requests.RequestException as e:
        logger.warning(f"ESPN JSON profile failed for {player_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# 5. Generic article scraper — unchanged
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
    Does NOT support paywalled content.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        for tag in soup.find_all(["script", "style", "nav", "footer",
                                   "header", "aside", "iframe", "noscript"]):
            tag.decompose()

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

        if not content_text:
            paragraphs = soup.find_all("p")
            content_text = "\n".join(
                p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40
            )

        title = soup.title.string.strip() if soup.title and soup.title.string else url
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
