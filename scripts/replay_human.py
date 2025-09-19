#!/usr/bin/env python3
import argparse
import tarfile
from bson import decode_file_iter
from typing import Optional
from pathlib import Path

import os
# Reduce pygame banner
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

from pygame.locals import K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT

# Use repo utils to load VGDL envs from text files
import sys as _sys
_sys.path.append(str(Path(__file__).resolve().parents[1]))
from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings
from vgdl.core import VGDLSprite
import pygame
import numpy as np
import imageio
import subprocess

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
    # Ensure output directory exists and frames are valid
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [_ensure_even_dims(fr) for fr in frames]
    # Write via imageio-ffmpeg (libx264 + yuv420p for compatibility)
    with imageio.get_writer(
        str(out_path), fps=fps, codec='libx264', format='FFMPEG', pixelformat='yuv420p'
    ) as w:
        for fr in frames:
            w.append_data(fr)
    return

def replay_one(game_name: str, level_idx: int, actions, games_dir: Path, max_steps: int = 500, visualize: bool=False, delay_ms: int=100, gif_path: Optional[Path]=None, mp4_path: Optional[Path]=None):
    game_path = games_dir / f"{game_name}.txt"
    level_path = games_dir / f"{game_name}_lvl{level_idx}.txt"
    if not game_path.exists() or not level_path.exists():
        raise FileNotFoundError(f"Missing game or level file: {game_path}, {level_path}")
    game_str = game_path.read_text()
    level_str = level_path.read_text()
    rle = createRLInputGameFromStrings(game_str, level_str)
    frames = [] if (gif_path or mp4_path) else None
    if visualize or frames is not None:
        rle.visualize = True
        # Initialize a real display (headless vars should not be set by caller)
        rle._game._initScreen(rle._game.screensize, headless=False)
        pygame.display.flip()
        # draw initial frame
        rle._game._drawAll()
        pygame.display.update(VGDLSprite.dirtyrects)
        VGDLSprite.dirtyrects = []
        if frames is not None:
            surf = pygame.surfarray.array3d(rle._game.screen)
            frames.append(np.transpose(surf, (1,0,2)))
    steps = 0
    ended = False
    end_info = {'win': False, 'termination': None}
    for a, ts in actions:
        key = ACTION_MAP.get(str(a).lower())
        if key is None:
            # ignore unknowns
            continue
        res = rle.step(key, return_obs=False)
        if visualize or frames is not None:
            rle._game._drawAll()
            pygame.display.update(VGDLSprite.dirtyrects)
            VGDLSprite.dirtyrects = []
            if frames is not None:
                surf = pygame.surfarray.array3d(rle._game.screen)
                frames.append(np.transpose(surf, (1,0,2)))
            if visualize and delay_ms:
                pygame.time.wait(delay_ms)
        steps += 1
        if res.get('ended'):
            ended = True
            end_info = {'win': res.get('win'), 'termination': res.get('termination')}
            break
        if steps >= max_steps:
            break
    # Write GIF if requested
    if frames is not None:
        if gif_path is not None:
            imageio.mimsave(str(gif_path), frames, duration=max(0.01, delay_ms/1000.0))
        if mp4_path is not None:
            fps = max(1, int(round(1000.0 / max(1, delay_ms))))
            _write_mp4(frames, fps=fps, out_path=mp4_path)
    # Return termination info if we ended early; else default
    if ended:
        return {'steps': steps, **end_info}
    return {'steps': steps, 'win': False, 'termination': None}

def main():
    ap = argparse.ArgumentParser(description='Replay a few human plays offline from ds004323 dump.tar.gz')
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--game-name', default=None, help='e.g., vgfmri3_sokoban; default: first with non-empty actions')
    ap.add_argument('--level-idx', type=int, default=0)
    ap.add_argument('--limit', type=int, default=1, help='How many plays to replay')
    ap.add_argument('--offset', type=int, default=0, help='Skip this many matching plays before replaying')
    ap.add_argument('--visualize', action='store_true', help='Open a pygame window and render frames')
    ap.add_argument('--delay-ms', type=int, default=120, help='Delay between rendered steps')
    ap.add_argument('--gif-out', type=Path, default=None, help='Write an animated GIF to this path')
    ap.add_argument('--mp4-out', type=Path, default=None, help='Write an MP4 video to this path')
    args = ap.parse_args()

    root = Path(args.dataset_root)
    dump_tgz = root / 'behavior' / 'dump.tar.gz'
    games_dir = root / 'games'

    count = 0
    skipped = 0
    # Configure headless vs visualized SDL before any display init
    if args.visualize:
        # Ensure we are not forcing dummy drivers
        for var in ('SDL_VIDEODRIVER', 'SDL_AUDIODRIVER'):
            if var in os.environ and os.environ[var] == 'dummy':
                del os.environ[var]
    else:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

    for doc in iter_plays(dump_tgz):
        gname = doc.get('game_name') or doc.get('game')
        acts = doc.get('actions') or doc.get('keypresses') or []
        if not acts:
            continue
        if args.game_name and gname != args.game_name:
            continue
        if skipped < args.offset:
            skipped += 1
            continue

        print(f"Replaying subj={doc.get('subj_id')} run={doc.get('run_id')} game={gname} len(actions)={len(acts)}")
        try:
            result = replay_one(
                gname,
                args.level_idx,
                acts, games_dir,
                visualize=args.visualize,
                max_steps=500,
                delay_ms=args.delay_ms,
                gif_path=args.gif_out,
                mp4_path=args.mp4_out
            )
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
