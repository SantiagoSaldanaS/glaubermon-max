"""Automated High-Elo Showdown Replay Harvester."""

import os
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional


class ReplayHarvester:
    """Fetches high-Elo competitive battle replays from Pokémon Showdown."""

    def __init__(self, output_dir: str = "data/replays", min_rating: int = 1400):
        self.output_dir = output_dir
        self.min_rating = min_rating
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GlaubermonAI/1.0"}
        os.makedirs(self.output_dir, exist_ok=True)

    def search_replays(self, format_id: str = "gen9ou", max_pages: int = 10) -> List[Dict]:
        """Search replay index for high-rated battle IDs."""
        found_replays = []
        print(f"Searching Showdown replays for format '{format_id}' (target rating >= {self.min_rating})...")

        for page in range(1, max_pages + 1):
            url = f"https://replay.pokemonshowdown.com/search.json?format={format_id}&page={page}"
            try:
                resp = requests.get(url, headers=self.headers, timeout=10)
                if resp.status_code != 200:
                    break
                items = resp.json()
                if not items:
                    break

                for r in items:
                    rating = r.get("rating")
                    if rating is not None and rating >= self.min_rating:
                        found_replays.append(r)
                time.sleep(0.2)  # Respect rate limits
            except Exception as e:
                print(f"Error querying page {page}: {e}")
                break

        print(f"Found {len(found_replays)} qualifying replays (Rating >= {self.min_rating}).")
        return found_replays

    def download_replay(self, replay_id: str) -> Optional[Dict]:
        """Download individual replay log JSON."""
        cache_file = os.path.join(self.output_dir, f"{replay_id}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        url = f"https://replay.pokemonshowdown.com/{replay_id}.json"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                return data
        except Exception as e:
            print(f"Failed to download {replay_id}: {e}")
        return None

    def harvest(self, format_id: str = "gen9ou", max_pages: int = 10, max_workers: int = 8) -> List[str]:
        """Orchestrate search and parallel download of replays."""
        metadata_list = self.search_replays(format_id=format_id, max_pages=max_pages)
        replay_ids = [m["id"] for m in metadata_list]

        downloaded_paths = []
        print(f"Downloading {len(replay_ids)} replays with {max_workers} worker threads...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(self.download_replay, rid): rid for rid in replay_ids}
            for future in as_completed(future_to_id):
                rid = future_to_id[future]
                res = future.result()
                if res:
                    downloaded_paths.append(os.path.join(self.output_dir, f"{rid}.json"))

        print(f"Successfully harvested {len(downloaded_paths)} replays in '{self.output_dir}'.")
        return downloaded_paths


if __name__ == "__main__":
    harvester = ReplayHarvester(min_rating=1350)
    harvester.harvest(max_pages=15)
