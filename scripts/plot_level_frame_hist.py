#!/usr/bin/env python3
"""
Plot per-game histograms of total frames per level from MongoDB plays.

Definition here:
- "frames" = number of action steps recorded in a play (len(actions)).
- For each game and for each level index, we sum frames over all matching plays.

Output:
- Figure 1: one subplot per game (num_games x 1 grid), each a bar chart over
  level indices with bar height = total frames recorded for that level in the
  DB (optionally filtered by subject).
- Figure 2 (optional): one subplot per game with bar height = number of unique
  players (subjects) who played that level. Enabled via --out-players.

Examples:
  # All subjects, auto-detected games
  python scripts/plot_level_frame_hist.py --mongo-uri mongodb://localhost:27017 --db heroku_7lzprs54 \
      --out plots/level_frames_all.png

  # Single subject and a subset of games
  python scripts/plot_level_frame_hist.py --mongo-uri mongodb://localhost:27017 --db heroku_7lzprs54 \
      --subj-id 2 --games vgfmri3_sokoban vgfmri3_zelda --out plots/level_frames_subj2.png
"""
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import os
import sys as _sys
import numpy as np
import matplotlib.pyplot as plt


def _coerce_int(val: Union[str, int, float, None]) -> Optional[int]:
    if val is None:
        return None
    try:
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, float):
            return int(val)
        s = str(val).strip()
        if s == '':
            return None
        if '.' in s:
            return int(float(s))
        return int(s)
    except Exception:
        return None


def _normalize_level_str(s: str) -> str:
    return '\n'.join([line.rstrip() for line in str(s).strip().splitlines()])


def resolve_level_index_for_play(p: Dict[str, Any], levels_from_db: Optional[List[str]]) -> Optional[int]:
    # Prefer numeric field: level_id only
    lv = _coerce_int(p.get('level_id'))
    if lv is not None:
        return lv
    # Try matching embedded level_str to games.levels list if provided
    lv_str = p.get('level_str')
    if isinstance(lv_str, str) and levels_from_db:
        norm_levels = [_normalize_level_str(s) for s in levels_from_db]
        try:
            return norm_levels.index(_normalize_level_str(lv_str))
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser(description='Plot per-game histograms of total frames per level from MongoDB plays')
    ap.add_argument('--mongo-uri', default='mongodb://localhost:27017')
    ap.add_argument('--db', required=True, help='Database name')
    ap.add_argument('--subj-id', type=int, default=None, help='Optional subject filter (uses subj_id string match)')
    ap.add_argument('--games', nargs='*', default=None, help='Optional list of game_name values to include')
    ap.add_argument('--out', type=Path, default=None, help='Output image path for frames figure (.png). If omitted, shows window')
    ap.add_argument('--out-players', type=Path, default=None, help='Also write players-per-level figure to this path')
    ap.add_argument('--debug', action='store_true', help='Print debug summaries')
    args = ap.parse_args()

    # Connect to Mongo
    from pymongo import MongoClient
    client = MongoClient(args.mongo_uri)
    db = client[args.db]

    # Determine games list
    if args.games:
        game_names = list(args.games)
    else:
        try:
            # Prefer distinct from plays
            game_names = sorted(db.plays.distinct('game_name'))
        except Exception:
            game_names = []
    if not game_names:
        print('No games discovered. Provide --games.')
        return

    # Prepare figure for frames
    n = len(game_names)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3, 2.4*n)), sharex=False)
    if n == 1:
        axes = [axes]

    # Per game aggregation
    for ax, gname in zip(axes, game_names):
        # levels reference from db.games (optional)
        gdoc = db.games.find_one({'name': gname}) or {}
        levels_ref: List[str] = gdoc.get('levels') or []
        n_levels = len(levels_ref) if levels_ref else None

        # Build query
        q: Dict[str, Any] = {'game_name': gname, 'actions': {'$type': 'array', '$ne': []}}
        if args.subj_id is not None:
            q['subj_id'] = str(args.subj_id)

        # Pull plays
        cur = db.plays.find(q, projection={
            'actions': 1,
            'level_id': 1, 'level_str': 1,
            'subj_id': 1,
        })
        plays = list(cur)
        if args.debug:
            print(f'{gname}: docs_with_actions={len(plays)}')

        # Aggregate frames per level and unique players per level
        level_to_frames: Dict[int, int] = {}
        level_to_subjects: Dict[int, set] = {}
        unknown_frames = 0
        unknown_player_subjects: set = set()
        for p in plays:
            lv = resolve_level_index_for_play(p, levels_ref)
            frames = len(p.get('actions') or [])
            subj = p.get('subj_id')
            subj_norm = str(subj) if subj is not None else None
            if lv is None:
                unknown_frames += frames
                if subj_norm is not None:
                    unknown_player_subjects.add(subj_norm)
                continue
            level_to_frames[lv] = level_to_frames.get(lv, 0) + frames
            if subj_norm is not None:
                level_to_subjects.setdefault(lv, set()).add(subj_norm)

        # Decide x-axis levels
        if n_levels is None:
            # derive from observed levels
            observed_levels = level_to_frames.keys()
            n_levels = (max(observed_levels)+1) if observed_levels else 1
        x = np.arange(n_levels)
        y = np.array([level_to_frames.get(i, 0) for i in range(n_levels)], dtype=float)

        ax.bar(x, y, color='#1f77b4', alpha=0.85)
        ax.set_title(f'{gname} — total frames per level' + (f' (subj {args.subj_id})' if args.subj_id is not None else ''))
        ax.set_ylabel('frames')
        ax.set_xlabel('level index')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{i}' for i in x])

        # Annotate bar values for readability (sparse labels)
        for i, val in enumerate(y):
            if val > 0:
                ax.text(i, val, f'{int(val)}', ha='center', va='bottom', fontsize=8, rotation=0)

        if args.debug and (unknown_frames or unknown_player_subjects):
            print(f'  unknown_level_frames={unknown_frames}, unknown_level_unique_players={len(unknown_player_subjects)}')

    plt.tight_layout()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(args.out), dpi=150)
        print('Wrote', args.out)
    else:
        plt.show()

    # Optional second figure: players per level
    if args.out_players is not None:
        fig2, axes2 = plt.subplots(n, 1, figsize=(12, max(3, 2.4*n)), sharex=False)
        if n == 1:
            axes2 = [axes2]

        # Recompute per game (could cache above, but keep simple/clear)
        for ax2, gname in zip(axes2, game_names):
            gdoc = db.games.find_one({'name': gname}) or {}
            levels_ref: List[str] = gdoc.get('levels') or []
            n_levels = len(levels_ref) if levels_ref else None

            q: Dict[str, Any] = {'game_name': gname, 'actions': {'$type': 'array', '$ne': []}}
            if args.subj_id is not None:
                q['subj_id'] = str(args.subj_id)

            cur = db.plays.find(q, projection={
                'actions': 1,
                'level_id': 1, 'level_str': 1,
                'subj_id': 1,
            })
            plays = list(cur)

            level_to_subjects: Dict[int, set] = {}
            for p in plays:
                lv = resolve_level_index_for_play(p, levels_ref)
                subj = p.get('subj_id')
                if lv is None or subj is None:
                    continue
                level_to_subjects.setdefault(lv, set()).add(str(subj))

            if n_levels is None:
                observed = level_to_subjects.keys()
                n_levels = (max(observed)+1) if observed else 1
            x = np.arange(n_levels)
            y = np.array([len(level_to_subjects.get(i, set())) for i in range(n_levels)], dtype=float)

            ax2.bar(x, y, color='#ff7f0e', alpha=0.85)
            title_suffix = f' (subj {args.subj_id})' if args.subj_id is not None else ''
            ax2.set_title(f'{gname} — unique players per level{title_suffix}')
            ax2.set_ylabel('players')
            ax2.set_xlabel('level index')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'L{i}' for i in x])
            for i, val in enumerate(y):
                if val > 0:
                    ax2.text(i, val, f'{int(val)}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        args.out_players.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(args.out_players), dpi=150)
        print('Wrote', args.out_players)


if __name__ == '__main__':
    main()
