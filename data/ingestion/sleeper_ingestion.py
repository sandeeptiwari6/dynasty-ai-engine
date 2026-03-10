import requests
from typing import Dict, List, Any
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd

from data.schema.database import Player, SleeperLeagueSnapshot, get_session

logger = logging.getLogger(__name__)

class SleeperClient:
    """
    Thin wrapper around Sleeper's public REST API.
    Docs: https://docs.sleeper.com/
    """

    BASE_URL = "https://api.sleeper.app/v1"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def _get(self, endpoint: str) -> Any:
        url = Path(self.BASE_URL) / endpoint
        response = requests.get(url, timeout=self.timeout)

        if response.status_code != 200:
            raise Exception(
                f"Request failed: {response.status_code} - {response.text}"
            )

        return response.json()

    # ---------------------------
    # League Endpoints
    # ---------------------------

    def get_league(self, league_id: str) -> Dict:
        return self._get(f"league/{league_id}")

    def get_league_users(self, league_id: str) -> List[Dict]:
        return self._get(f"league/{league_id}/users")

    def get_league_rosters(self, league_id: str) -> List[Dict]:
        return self._get(f"league/{league_id}/rosters")

    def get_league_matchups(self, league_id: str, week: int) -> List[Dict]:
        return self._get(f"league/{league_id}/matchups/{week}")

    def get_league_transactions(self, league_id: str, week: int) -> List[Dict]:
        return self._get(f"league/{league_id}/transactions/{week}")
    
    def get_league_traded_picks(self, league_id: str) -> List[Dict]:
        return self._get(f"league/{league_id}/traded_picks")
    

    # ---------------------------
    # Draft Endpoints
    # ---------------------------

    def get_league_drafts(self, league_id: str) -> List[Dict]:
        return self._get(f"league/{league_id}/drafts")
    
    def get_user_drafts(self, user_id: str, season: int) -> List[Dict]:
        return self._get(f"user/{user_id}/drafts/nfl/{season}")

    def get_draft(self, draft_id: str) -> Dict:
        return self._get(f"draft/{draft_id}")
    
    def get_draft_picks(self, draft_id: str) -> List[Dict]:
        return self._get(f"draft/{draft_id}/picks")

    def get_draft_traded_picks(self, draft_id: str) -> List[Dict]:
        return self._get(f"draft/{draft_id}/traded_picks")

    def get_nfl_state(self) -> Dict:
        return self._get(f"state/nfl")
    
    # ---------------------------
    # Global NFL Data
    # ---------------------------

    def get_all_players(self) -> Dict:
        """
        Returns the full Sleeper player universe as a dict keyed by sleeper_id.

        WARNING: Large payload (~10-15MB).
        Cache locally after first pull.
        """
        return self._get("players/nfl")

class SleeperIngestion:
    """
    Syncs Sleeper player data and league context into SQLite.

    Usage:
        pipeline = SleeperIngestionPipeline(engine)
        pipeline.ingest_players()
        pipeline.ingest_trending_snapshot()
        pipeline.ingest_league(league_id="your_league_id")  # optional
    """

    SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

    def __init__(self, engine):
        self.engine = engine
        self.client = SleeperClient()
    
    def ingest_players(self):
        """
        Pull full player universe from Sleeper and:
        1. Create/update Player records in our DB (for players not already there from nfl_data_py)
        2. Update sleeper_id on existing players to enable cross-source joining
        """

        logger.info("Ingesting Sleeper player universe...")
        raw_player_data = self.client.get_all_players()

        updated = 0
        created = 0

        with get_session(self.engine) as session:
            for sleeper_id, player_data in raw_player_data.items():
                pos = player_data.get("position")

                if pos in self.SKILL_POSITIONS:
                    gsis_id = player_data.get("player_id") or player_data.get("gsis_id")

                    existing = None
                    if gsis_id:
                        existing = session.get(Player, str(gsis_id))

                    if existing:
                        # Update Sleeper ID on existing record
                        existing.sleeper_id = sleeper_id
                        updated += 1
                    else:
                        # Create minimal player record (will be enriched by NFL ingestion)
                        name = player_data.get("full_name") or f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}".strip()
                        player = Player(
                            player_id=gsis_id or sleeper_id,
                            gsis_id=gsis_id,
                            sleeper_id=sleeper_id,
                            name=name,
                            first_name=player_data.get("first_name"),
                            last_name=player_data.get("last_name"),
                            position=pos,
                            nfl_team=player_data.get("team"),
                            status=player_data.get("status"),
                            years_exp=player_data.get("years_exp"),
                            depth_chart_order=player_data.get("depth_chart_order"),
                            jersey_number=player_data.get("number"),
                            college_team=player_data.get("college"),
                            updated_at=datetime.utcnow(),
                        )
                        session.merge(player)
                        created += 1

            session.commit()

        logger.info(f"  → Sleeper universe: {updated} updated, {created} created.")
    
    def ingest_league(self, league_id: str):
        """
        Pull a user's specific dynasty league context.
        Returns a parsed structure useful for the dynasty advisor agent.

        Returns:
            {
                "league_info": {...},
                "rosters": [{"owner": ..., "players": [...]}],
                "roster_counts": {"QB": 2, "RB": 5, ...}
            }
        """

        logger.info(f"Loading dynasty league {league_id}...")

        league_info = self.client.get_league(league_id)
        raw_roster_data = self.client.get_league_rosters(league_id)
        raw_user_data = self.client.get_league_users(league_id)

        user_map = {u["user_id"]: u.get("display_name", "Unknown") for u in raw_user_data}

        rosters = []
        for roster in raw_roster_data:
            owner = user_map.get(roster.get("owner_id"), "Unknown")

            player_ids = roster.get("players", [])
            players_data = []

            with get_session(self.engine) as session:
                for player_id in player_ids:
                    player = session.query(Player).filter(
                        Player.sleeper_id == player_id
                    ).first()

                    if player:
                        player_data = {
                            "name": player.name,
                            "position": player.position,
                            "team": player.nfl_team,
                            "years_exp": player.years_exp,
                            "sleeper_id": player_id,
                        }
                    else:
                        player_data = {
                            "sleeper_id": player_id, 
                            "name": "Unknown"
                        }
                    players_data.append(player_data)
            rosters.append(
                {
                    "owner": owner,
                    "roster_id": roster.get("roster_id"),
                    "players": players_data,
                    "picks": roster.get("draft_picks", []),
                }
            )

        # Compute position counts across league (helps identify scarcity)
        position_counts = {}
        for roster in rosters:
            for player in roster["players"]:
                if player.get("position"):
                    pos = player["position"]
                    position_counts[pos] = position_counts.get(pos, 0) + 1
        
        league_data = {
            "league_info": league_info,
            "league_name": league_info.get("name"),
            "scoring_settings": league_info.get("scoring_settings", {}),
            "roster_positions": league_info.get("roster_positions", []),
            "rosters": rosters,
            "league_position_counts": position_counts,
        }

        logger.info(f"  → League '{league_data['league_name']}' loaded with {len(rosters)} teams.")
        return league_data
    
    def get_player_adp(self):
        """
        Returns a DataFrame of all skill position players with
        Sleeper trending add counts as a proxy for dynasty ADP sentiment.
        Useful for dynasty value ranking in the ML model.
        """

        raw_player_data = self.client.get_all_players()
        adp_data = []

        for sleeper_id, player_data in raw_player_data.items():
            if player_data.get("position") in self.SKILL_POSITIONS:
                adp_data.append(
                    {
                        "sleeper_id": sleeper_id,
                        "name": player_data.get("full_name"),
                        "position": player_data.get("position"),
                        "team": player_data.get("team"),
                        "age": player_data.get("age"),
                        "years_exp": player_data.get("years_exp"),
                        "status": player_data.get("status"),
                        "depth_chart_order": player_data.get("depth_chart_order"),
                        "injury_status": player_data.get("injury_status"),
                    }
                )
        return pd.DataFrame(adp_data)