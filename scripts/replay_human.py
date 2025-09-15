#!/usr/bin/env python3
import argparse
import tarfile
from bson import decode_file_iter
from typing import Optional
from pathlib import Path

import os
# Ensure fully headless pygame init in CI/sandbox environments
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from pygame.locals import K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT

# Use repo utils to load VGDL envs from text files
import sys as _sys
_sys.path.append(str(Path(__file__).resolve().parents[1]))
from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings

ACTION_MAP = {
    'spacebar': K_SPACE,
    'up': K_UP,
    'down': K_DOWN,
    'left': K_LEFT,
    'right': K_RIGHT,
}

def find_member(tf: tarfile.TarFile, endswith: str) -> Optional[tarfile.TarInfo]:
    for m in tf.getmembers():
        if m.name.endswith(endswith):
            return m
    return None

def iter_plays(dump_tgz: Path):
    with tarfile.open(dump_tgz, mode='r:gz') as tf:
        m = find_member(tf, '/plays.bson') or find_member(tf, 'plays.bson')
        if not m:
            raise FileNotFoundError('plays.bson not found inside dump tar.gz')
        f = tf.extractfile(m)
        for doc in decode_file_iter(f):
            yield doc

def replay_one(game_name: str, level_idx: int, actions, games_dir: Path, max_steps: int = 500):
    game_path = games_dir / f"{game_name}.txt"
    level_path = games_dir / f"{game_name}_lvl{level_idx}.txt"
    if not game_path.exists() or not level_path.exists():
        raise FileNotFoundError(f"Missing game or level file: {game_path}, {level_path}")
    game_str = game_path.read_text()
    level_str = level_path.read_text()
    rle = createRLInputGameFromStrings(game_str, level_str)
    steps = 0
    for a, ts in actions:
        key = ACTION_MAP.get(str(a).lower())
        if key is None:
            # ignore unknowns
            continue
        res = rle.step(key, return_obs=False)
        steps += 1
        if res.get('ended'):
            return {'steps': steps, 'win': res.get('win'), 'termination': res.get('termination')}
        if steps >= max_steps:
            break
    return {'steps': steps, 'win': False, 'termination': None}

def main():
    ap = argparse.ArgumentParser(description='Replay a few human plays offline from ds004323 dump.tar.gz')
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--game-name', default=None, help='e.g., vgfmri3_sokoban; default: first with non-empty actions')
    ap.add_argument('--level-idx', type=int, default=0)
    ap.add_argument('--limit', type=int, default=1, help='How many plays to replay')
    args = ap.parse_args()

    root = Path(args.dataset_root)
    dump_tgz = root / 'behavior' / 'dump.tar.gz'
    games_dir = root / 'games'

    count = 0
    for doc in iter_plays(dump_tgz):
        gname = doc.get('game_name') or doc.get('game')
        acts = doc.get('actions') or doc.get('keypresses') or []
        if not acts:
            continue
        if args.game_name and gname != args.game_name:
            continue

        print(f"Replaying subj={doc.get('subj_id')} run={doc.get('run_id')} game={gname} len(actions)={len(acts)}")
        try:
            result = replay_one(gname, args.level_idx, acts, games_dir)
            print('Result:', result)
        except Exception as e:
            print('Replay error:', e)
        count += 1
        if count >= args.limit:
            break

    if count == 0:
        print('No matching plays with actions found.')

if __name__ == '__main__':
    main()
