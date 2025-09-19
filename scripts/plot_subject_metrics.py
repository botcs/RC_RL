#!/usr/bin/env python3
"""
Plot per-subject per-game episode metrics from MongoDB plays.

For a given subject, this script:
- Flattens plays across runs (sorted by time)
- Computes per-play steps and total reward by replaying actions
- Draws vertical delimiters for each level index
- Puts all games in a [num_games x 1] grid of subplots
- Annotates/overlays max steps and max reward per game

Usage example:
  python scripts/plot_subject_metrics.py \
    --mongo-uri mongodb://localhost:27017 --db vgfmri \
    --subj-id 2 --games-source db --out plots/subj2_metrics.png

Notes:
- Uses `db.games` to fetch game/level strings by default. You can alternatively
  load game/level texts from dataset files with --games-source files and
  --dataset-root pointing to the ds004323 folder containing `games/`.
"""
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import os
import sys as _sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Reduce pygame banner and ensure headless for replay computations
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

# Ensure repo root is on path
_sys.path.append(str(Path(__file__).resolve().parents[1]))

from pygame.locals import K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT
from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings


ACTION_MAP = {
    'spacebar': K_SPACE,
    'up': K_UP,
    'down': K_DOWN,
    'left': K_LEFT,
    'right': K_RIGHT,
}


def get_game_and_level_strings_db(db, game_name: str, level_idx: int, cache: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    if game_name not in cache:
        doc = db.games.find_one({'name': game_name})
        if not doc:
            raise RuntimeError(f'Game {game_name} not found in DB.games')
        cache[game_name] = doc
    doc = cache[game_name]
    descs = doc.get('descs')
    levels = doc.get('levels')
    if not isinstance(descs, list) or not isinstance(levels, list):
        raise RuntimeError(f'Unexpected schema in DB.games for {game_name}: expected lists in descs/levels')
    try:
        return descs[level_idx], levels[level_idx]
    except Exception:
        raise IndexError(f'Level index {level_idx} out of range for game {game_name}')


def get_game_and_level_strings_files(root: Path, game_name: str, level_idx: int) -> Tuple[str, str]:
    games_dir = root / 'games'
    gp = games_dir / f'{game_name}.txt'
    lp = games_dir / f'{game_name}_lvl{level_idx}.txt'
    if not gp.exists() or not lp.exists():
        raise FileNotFoundError(f'Missing game or level file: {gp}, {lp}')
    return gp.read_text(), lp.read_text()


def _coerce_int(val: Union[str, int, float, None]) -> Optional[int]:
    if val is None:
        return None
    # Already an int-like
    try:
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, float):
            return int(val)
        # string/other representations
        s = str(val).strip()
        if s == '':
            return None
        # Accept forms like '0', '2', '3.0'
        if '.' in s:
            return int(float(s))
        return int(s)
    except Exception:
        return None


def _normalize_level_str(s: str) -> str:
    # Normalize whitespace/newlines for robust equality
    return '\n'.join([line.rstrip() for line in str(s).strip().splitlines()])


def resolve_game_and_level_for_play(
    p: Dict[str, Any],
    gname: str,
    games_source: str,
    db,
    games_cache: Dict[str, Dict[str, Any]],
    dataset_root: Optional[Path]
) -> Tuple[str, str, Optional[int]]:
    """
    Robustly resolve (game_str, level_str, level_idx) for a play document.
    - Prefer numeric level index if present under common field names.
    - If a full level string is embedded in the play doc, use it directly and
      attempt to back-map it to its index by matching against DB/files.
    """
    # 1) Try direct numeric level index using level_id only (dataset convention)
    level_idx = _coerce_int(p.get('level_id'))

    embedded_level_str = None
    # If level could be a string description (ASCII map), capture it
    # Some databases also include full level text under level_str
    lv = p.get('level_str')
    if isinstance(lv, str) and ('\n' in lv or len(lv) > 50):
        embedded_level_str = _normalize_level_str(lv)

    # 2) Obtain game_str and level_str
    game_str: str
    level_str: str

    if games_source == 'db':
        # Fetch game doc to get game/level strings
        if gname not in games_cache:
            doc = db.games.find_one({'name': gname})
            if not doc:
                raise RuntimeError(f'Game {gname} not found in DB.games')
            games_cache[gname] = doc
        gdoc = games_cache[gname]
        descs = gdoc.get('descs') or []
        levels = gdoc.get('levels') or []
        if not descs or not levels:
            raise RuntimeError(f'Game doc for {gname} missing descs/levels')

        # Pick game_str: prefer index if available, else first
        if level_idx is not None and 0 <= level_idx < len(descs):
            game_str = descs[level_idx]
        else:
            game_str = descs[0]

        # Level string resolution
        if level_idx is not None and 0 <= level_idx < len(levels):
            level_str = levels[level_idx]
        elif embedded_level_str is not None:
            # Try to match embedded string to one of gdoc.levels
            norm_levels = [_normalize_level_str(s) for s in levels]
            try:
                idx = norm_levels.index(_normalize_level_str(embedded_level_str))
                level_idx = idx
                level_str = levels[idx]
            except Exception:
                # Fallback: use embedded string directly
                level_str = embedded_level_str
        else:
            # As a last resort, default to level 0
            level_idx = level_idx if level_idx is not None else 0
            if 0 <= level_idx < len(levels):
                level_str = levels[level_idx]
            else:
                level_str = levels[0]

    else:
        # games_source == 'files'
        if not dataset_root:
            raise RuntimeError('Provide --dataset-root when using --games-source files')
        gp = (dataset_root / 'games' / f'{gname}.txt')
        if not gp.exists():
            raise FileNotFoundError(f'Missing game file: {gp}')
        game_str = gp.read_text()

        if level_idx is not None:
            # Use explicit index
            _, level_str = get_game_and_level_strings_files(dataset_root, gname, level_idx)
        elif embedded_level_str is not None:
            # Try to back-map to index by comparing to files
            # Search for first level file that matches; otherwise, use embedded
            games_dir = dataset_root / 'games'
            matched = False
            i = 0
            while True:
                lp = games_dir / f'{gname}_lvl{i}.txt'
                if not lp.exists():
                    break
                try:
                    txt = _normalize_level_str(lp.read_text())
                    if txt == embedded_level_str:
                        level_idx = i
                        level_str = lp.read_text()
                        matched = True
                        break
                except Exception:
                    pass
                i += 1
            if not matched:
                level_str = embedded_level_str
        else:
            # Fallback to level 0
            level_idx = 0
            _, level_str = get_game_and_level_strings_files(dataset_root, gname, 0)

    return game_str, level_str, level_idx


def compute_steps_and_reward(game_str: str, level_str: str, actions: List) -> Tuple[int, float]:
    """Replay actions headlessly and compute (steps, total_reward)."""
    rle = createRLInputGameFromStrings(game_str, level_str)
    # Ensure headless stepping
    rle.visualize = False
    total_reward = 0.0
    steps = 0
    for a, _ts in actions:
        key = ACTION_MAP.get(str(a).lower())
        if key is None:
            continue
        res = rle.step(key, return_obs=False)
        total_reward += float(res.get('reward', 0.0))
        steps += 1
        if res.get('ended'):
            break
    return steps, total_reward


def main():
    ap = argparse.ArgumentParser(description='Plot per-subject metrics (steps, reward) per game with level delimiters')
    ap.add_argument('--mongo-uri', default='mongodb://localhost:27017')
    ap.add_argument('--db', required=True, help='Database name containing plays/games collections')
    ap.add_argument('--subj-id', type=int, required=True)
    ap.add_argument('--games-source', choices=['db', 'files'], default='db')
    ap.add_argument('--dataset-root', default=None, help='Path to ds004323 root (required if --games-source files)')
    ap.add_argument('--games', nargs='*', default=None, help='Optional list of game_name values to include')
    ap.add_argument('--limit-per-game', type=int, default=0, help='Optional limit of plays per game (0=all)')
    ap.add_argument('--out', type=Path, default=None, help='Output image path (.png). If omitted, calls plt.show().')
    ap.add_argument('--debug', action='store_true', help='Print per-game summaries and sanity checks to stdout')
    ap.add_argument('--mark-runs', action='store_true', help='Draw vertical delimiters at run boundaries as well')
    args = ap.parse_args()

    # Connect to Mongo
    from pymongo import MongoClient
    client = MongoClient(args.mongo_uri)
    db = client[args.db]

    # Build base query: this subject only (DB stores subj_id as string), with non-empty actions
    q: Dict[str, Any] = {
        'subj_id': str(args.subj_id),
        'actions': {'$type': 'array', '$ne': []},
    }
    if args.games:
        q['game_name'] = {'$in': args.games}

    # Pull all plays for subject
    cur = db.plays.find(q, projection={
        'game_name': 1,
        'level_id': 1,
        'level_str': 1,
        'actions': 1,
        'start_time': 1,
        'run_id': 1,
    }).sort([('start_time', 1), ('run_id', 1)])
    plays = list(cur)
    if not plays:
        print('No plays found for subject', args.subj_id)
        return

    # Partition by game
    by_game: Dict[str, List[Dict[str, Any]]] = {}
    for p in plays:
        gname = p.get('game_name') or p.get('game')
        if not gname:
            # Skip malformed docs
            continue
        by_game.setdefault(gname, []).append(p)

    # Optionally cap per-game plays
    if args.limit_per_game and args.limit_per_game > 0:
        for g in list(by_game.keys()):
            by_game[g] = by_game[g][:args.limit_per_game]

    # Decide game order: historical (first time each game was played)
    def _as_ts(t) -> float:
        # Robust conversion to sortable float timestamp
        if t is None:
            return float('inf')
        try:
            if hasattr(t, 'timestamp'):
                return float(t.timestamp())
        except Exception:
            pass
        if isinstance(t, (int, float)):
            return float(t)
        if isinstance(t, str):
            # Try ISO 8601 without external deps
            try:
                return datetime.fromisoformat(t).timestamp()
            except Exception:
                return float('inf')
        return float('inf')

    def _first_ts(plays_for_game: List[Dict[str, Any]]) -> float:
        vals = [_as_ts(p.get('start_time')) for p in plays_for_game]
        return min(vals) if vals else float('inf')

    game_names = sorted(by_game.keys(), key=lambda g: _first_ts(by_game[g]))
    if not game_names:
        print('No valid game_name entries for subject', args.subj_id)
        return

    # Prepare plot grid
    n_games = len(game_names)
    fig, axes = plt.subplots(n_games, 1, figsize=(10, max(3, 3*n_games)), sharex=False)
    if n_games == 1:
        axes = [axes]

    # Optionally set up dataset root
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    games_cache: Dict[str, Dict[str, Any]] = {}

    for ax, gname in zip(axes, game_names):
        gplays = by_game[gname]

        # Sort plays by (run_id, start_time) then flatten; also record level boundaries
        def _lvl(p):
            return int(p.get('level_id', 0) or 0)
        gplays_sorted = sorted(gplays, key=lambda p: (p.get('run_id', 0), _as_ts(p.get('start_time'))))

        if args.debug:
            # Summarize levels and runs straight from raw fields (level_id only)
            lv_counts_raw: Dict[Optional[int], int] = {}
            run_counts: Dict[Any, int] = {}
            for p in gplays_sorted:
                raw_lv = p.get('level_id')
                try:
                    raw_lv = int(raw_lv) if raw_lv is not None and str(raw_lv) != '' else None
                except Exception:
                    raw_lv = None
                lv_counts_raw[raw_lv] = lv_counts_raw.get(raw_lv, 0) + 1
                run_counts[p.get('run_id', None)] = run_counts.get(p.get('run_id', None), 0) + 1
            print(f'Game {gname}: docs={len(gplays_sorted)}, raw_level_counts={lv_counts_raw}, run_counts={run_counts}')

        # Compute per-play metrics
        steps_list: List[int] = []
        reward_list: List[float] = []
        level_list: List[Optional[int]] = []

        docs_with_actions = 0
        for p in gplays_sorted:
            game_str, level_str, lvl_idx = resolve_game_and_level_for_play(
                p, gname, args.games_source, db, games_cache, dataset_root
            )
            level_list.append(lvl_idx)

            acts = p.get('actions') or p.get('keypresses') or []
            if not acts:
                steps_list.append(0)
                reward_list.append(0.0)
                continue
            docs_with_actions += 1
            s, r = compute_steps_and_reward(game_str, level_str, acts)
            steps_list.append(int(s))
            reward_list.append(float(r))

        x = np.arange(len(steps_list))

        # Plot steps and reward; use twin y-axis for readability
        color_steps = '#1f77b4'
        color_reward = '#d62728'
        ax2 = ax.twinx()
        ax.plot(x, steps_list, marker='o', linestyle='-', color=color_steps, label='steps', alpha=0.8)
        ax2.plot(x, reward_list, marker='^', linestyle='--', color=color_reward, label='reward', alpha=0.7)

        # Level delimiters: draw a vertical line at the start index of each new level
        level_changes = []
        prev = None
        for i, lvl in enumerate(level_list):
            if prev is None or lvl != prev:
                level_changes.append((i, lvl))
                prev = lvl
        for i, lvl in level_changes:
            ax.axvline(i-0.5, color='gray', linestyle=':', linewidth=1)
            label = f'L{lvl}' if lvl is not None else 'L?'
            ax.text(i, ax.get_ylim()[1]*0.95, label, color='gray', fontsize=8, ha='left', va='top', rotation=0)

        # Optionally mark run boundaries as well
        if args.mark_runs:
            prev_run = None
            for i, p in enumerate(gplays_sorted):
                r = p.get('run_id', None)
                if prev_run is None:
                    prev_run = r
                    continue
                if r != prev_run:
                    ax.axvline(i-0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
                    prev_run = r

        # Annotate max lines
        if steps_list:
            max_steps = max(steps_list)
            ax.axhline(max_steps, color=color_steps, linestyle=':', linewidth=1)
            ax.text(x[-1] if len(x) else 0, max_steps, f'max steps={max_steps}', color=color_steps, fontsize=8, ha='right', va='bottom')
        if reward_list:
            max_reward = max(reward_list)
            ax2.axhline(max_reward, color=color_reward, linestyle=':', linewidth=1)
            ax2.text(x[-1] if len(x) else 0, max_reward, f'max reward={max_reward:.2f}', color=color_reward, fontsize=8, ha='right', va='bottom')

        if args.debug:
            # post-resolve level counts
            counts_resolved: Dict[Optional[int], int] = {}
            for lv in level_list:
                counts_resolved[lv] = counts_resolved.get(lv, 0) + 1
            print(f'  resolved_levels={counts_resolved}, plays_used={docs_with_actions}')

        ax.set_title(f'{gname} — subject {args.subj_id} (n={len(steps_list)})')
        ax.set_ylabel('steps')
        ax2.set_ylabel('reward')
        ax.set_xlabel('play index (flattened)')

    plt.tight_layout()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(args.out), dpi=150)
        print('Wrote', args.out)
    else:
        plt.show()


if __name__ == '__main__':
    main()
