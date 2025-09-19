#!/usr/bin/env python3
"""
Replay human plays directly from a MongoDB dump (no tar streaming).

Requirements:
  - MongoDB running locally with the dataset restored (collections: plays, games, ...)
  - pymongo installed in your Python env

Usage examples:
  # Simple replay of first matching play (uses game files on disk):
  python scripts/replay_human_from_db.py \
      --mongo-uri mongodb://localhost:27017 --db vgfmri \
      --dataset-root ~/Downloads/ds004323-download \
      --game-name vgfmri3_sokoban --level-idx 0 --limit 1

  # Visualize
  python scripts/replay_human_from_db.py --visualize --delay-ms 80 ...

  # Export MP4
  python scripts/replay_human_from_db.py --mp4-out out.mp4 --delay-ms 120 ...

  # Use games stored in MongoDB instead of text files
  python scripts/replay_human_from_db.py --games-source db --db vgfmri ...
"""
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import os
import numpy as np
import imageio
import subprocess
import sys as _sys

from pygame.locals import K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT

# Ensure repo root is on path
_sys.path.append(str(Path(__file__).resolve().parents[1]))

from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings
from vgdl.core import VGDLSprite
import pygame

ACTION_MAP = {
    'spacebar': K_SPACE,
    'up': K_UP,
    'down': K_DOWN,
    'left': K_LEFT,
    'right': K_RIGHT,
}


def _ensure_even_dims(fr: np.ndarray) -> np.ndarray:
    fr = np.asarray(fr)
    h, w = fr.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        fr = np.pad(fr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    if fr.dtype != np.uint8:
        fr = fr.astype(np.uint8)
    return fr


def _write_mp4(frames, fps: int, out_path: Path) -> None:
    frames = [_ensure_even_dims(fr) for fr in frames]
    # Try imageio-ffmpeg first
    try:
        try:
            import imageio_ffmpeg  # noqa: F401
        except Exception:
            pass
        with imageio.get_writer(str(out_path), fps=fps, codec='libx264', format='FFMPEG', pixelformat='yuv420p') as w:
            for fr in frames:
                w.append_data(fr)
        return
    except Exception as e_primary:
        # Fallback to system ffmpeg
        try:
            import tempfile
            tmpdir = tempfile.mkdtemp(prefix='vgdl_mp4_')
            for i, fr in enumerate(frames):
                imageio.imwrite(os.path.join(tmpdir, f'frame_{i:06d}.png'), fr)
            cmd = [
                'ffmpeg', '-y', '-r', str(fps), '-i', os.path.join(tmpdir, 'frame_%06d.png'),
                '-movflags', 'faststart', '-pix_fmt', 'yuv420p', str(out_path)
            ]
            subprocess.run(cmd, check=True)
            return
        except Exception as e_fallback:
            raise RuntimeError(
                f"MP4 writer not available. Install 'imageio-ffmpeg' or 'ffmpeg' CLI. Primary error: {e_primary}; Fallback error: {e_fallback}"
            )


def replay_one(game_str: str, level_str: str, actions, visualize: bool = False, delay_ms: int = 100,
               max_steps: int = 500, gif_path: Optional[Path] = None, mp4_path: Optional[Path] = None) -> Dict[str, Any]:
    rle = createRLInputGameFromStrings(game_str, level_str)
    frames = [] if (gif_path or mp4_path) else None
    if visualize or frames is not None:
        rle.visualize = True
        rle._game._initScreen(rle._game.screensize, headless=False)
        pygame.display.flip()
        # initial frame
        rle._game._drawAll()
        pygame.display.update(VGDLSprite.dirtyrects)
        VGDLSprite.dirtyrects = []
        if frames is not None:
            surf = pygame.surfarray.array3d(rle._game.screen)
            frames.append(np.transpose(surf, (1, 0, 2)))

    steps = 0
    for a, ts in actions:
        key = ACTION_MAP.get(str(a).lower())
        if key is None:
            continue
        res = rle.step(key, return_obs=False)
        if visualize or frames is not None:
            rle._game._drawAll()
            pygame.display.update(VGDLSprite.dirtyrects)
            VGDLSprite.dirtyrects = []
            if frames is not None:
                surf = pygame.surfarray.array3d(rle._game.screen)
                frames.append(np.transpose(surf, (1, 0, 2)))
            if visualize and delay_ms:
                pygame.time.wait(delay_ms)
        steps += 1
        if res.get('ended'):
            result = {'steps': steps, 'win': res.get('win'), 'termination': res.get('termination')}
            break
        if steps >= max_steps:
            result = {'steps': steps, 'win': False, 'termination': None}
            break
    else:
        result = {'steps': steps, 'win': False, 'termination': None}

    # Write media
    if frames is not None:
        fps = max(1, int(round(1000.0 / max(1, delay_ms))))
        if gif_path is not None:
            try:
                imageio.mimsave(str(gif_path), frames, duration=max(0.01, delay_ms/1000.0))
            except Exception as e:
                print('GIF write error:', e)
        if mp4_path is not None:
            try:
                _write_mp4(frames, fps=fps, out_path=mp4_path)
            except Exception as e:
                print('MP4 write error:', e)
    return result


def main():
    ap = argparse.ArgumentParser(description='Replay human plays directly from MongoDB')
    ap.add_argument('--mongo-uri', default='mongodb://localhost:27017')
    ap.add_argument('--db', help='Database name (if omitted, tries to auto-detect by finding a DB with a plays collection)')
    ap.add_argument('--dataset-root', default=None, help='Path to ds004323 root (for game text files if not using DB)')
    ap.add_argument('--games-source', choices=['files', 'db'], default='files', help='Where to load game/level strings from')
    ap.add_argument('--game-name', default=None)
    ap.add_argument('--level-idx', type=int, default=0)
    ap.add_argument('--subj-id', type=int, default=None)
    ap.add_argument('--run-id', type=int, default=None)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--limit', type=int, default=1)
    ap.add_argument('--id', type=str, default=None, help='Replay a specific play by _id (takes precedence over other filters)')
    ap.add_argument('--visualize', action='store_true')
    ap.add_argument('--delay-ms', type=int, default=120)
    ap.add_argument('--gif-out', type=Path, default=None)
    ap.add_argument('--mp4-out', type=Path, default=None)
    args = ap.parse_args()

    # SDL headless vs visualize
    if args.visualize:
        for var in ('SDL_VIDEODRIVER', 'SDL_AUDIODRIVER'):
            if var in os.environ and os.environ[var] == 'dummy':
                del os.environ[var]
    else:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

    # Connect to Mongo
    from pymongo import MongoClient
    from bson import ObjectId
    client = MongoClient(args.mongo_uri)
    dbname = args.db
    if not dbname:
        # Auto-detect: pick first DB that has a 'plays' collection
        for d in client.list_database_names():
            if 'plays' in client[d].list_collection_names():
                dbname = d
                break
        if not dbname:
            raise RuntimeError('Could not auto-detect database with a plays collection. Pass --db explicitly.')
    db = client[dbname]

    # Build query
    q: Dict[str, Any] = {'actions': {'$type': 'array', '$ne': []}}

    # Specific document by _id (takes precedence)
    if args.id:
        try:
            q['_id'] = ObjectId(args.id)
        except Exception:
            # Fall back to string match if not a valid ObjectId
            q['_id'] = args.id
    if args.game_name and not args.id:
        q['game_name'] = args.game_name
    # Filter by subject and run using canonical fields (dataset stores subj_id as string, run_id as number)
    if args.subj_id is not None and not args.id:
        q['subj_id'] = str(args.subj_id)
    if args.run_id is not None and not args.id:
        q['run_id'] = args.run_id

    # Fast path: exact _id specified → fetch exactly one doc
    if args.id:
        p = db.plays.find_one(q)
        if not p:
            print('No document found for id', args.id)
            return
        docs = [p]
    else:
        cur = db.plays.find(q).sort('start_time', 1)
        if args.offset:
            cur = cur.skip(args.offset)
        if args.limit:
            cur = cur.limit(args.limit)
        docs = list(cur)

    # Resolve game + level strings
    games_dir = Path(args.dataset_root).joinpath('games') if args.dataset_root else None
    games_cache: Dict[str, Dict[str, Any]] = {}

    def get_game_and_level_strings(gname: str, level_idx: int) -> (str, str):
        if args.games_source == 'files':
            if not games_dir:
                raise RuntimeError('Provide --dataset-root when using --games-source files')
            gp = games_dir / f'{gname}.txt'
            lp = games_dir / f'{gname}_lvl{level_idx}.txt'
            if not gp.exists() or not lp.exists():
                raise FileNotFoundError(f'Missing game or level file: {gp}, {lp}')
            return gp.read_text(), lp.read_text()
        else:
            # games from DB
            if gname not in games_cache:
                doc = db.games.find_one({'name': gname})
                if not doc:
                    raise RuntimeError(f'Game {gname} not found in DB.games')
                games_cache[gname] = doc
            doc = games_cache[gname]
            # dataset schema: 'descs' (list of game strings), 'levels' (list of level strings)
            descs = doc.get('descs')
            levels = doc.get('levels')
            if not isinstance(descs, list) or not isinstance(levels, list):
                raise RuntimeError('Unexpected games doc schema; expected lists in descs/levels')
            try:
                return descs[level_idx], levels[level_idx]
            except Exception:
                raise IndexError(f'Level index {level_idx} out of range for game {gname}')

    played = 0
    for p in docs:
        gname = p.get('game_name') or p.get('game')
        acts = p.get('actions') or p.get('keypresses') or []
        if not acts:
            continue
        # Prefer level_id from DB plays; fall back to CLI level_idx
        lvl_idx = p.get('level_id') if p.get('level_id') is not None else args.level_idx
        pid = str(p.get('_id'))
        st = p.get('start_time')
        print(f"Replaying id={pid} subj={p.get('subj_id')} run={p.get('run_id')} game={gname} level_idx={lvl_idx} len(actions)={len(acts)} start={st}")

        game_str, level_str = get_game_and_level_strings(gname, lvl_idx)
        res = replay_one(game_str, level_str, acts, visualize=args.visualize, delay_ms=args.delay_ms,
                         gif_path=args.gif_out, mp4_path=args.mp4_out)
        print('Result:', res)
        played += 1

    if played == 0:
        print('No matching plays found.')


if __name__ == '__main__':
    main()
