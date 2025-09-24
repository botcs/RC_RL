# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a reinforcement learning research codebase focused on Atari-style VGDL games with fMRI data integration. The repository implements both Double Deep Q-Network (DDQN) and theory-based reinforcement learning agents (EMPA) for studying neural mechanisms of game learning.

## Key Architecture Components

### Core Game Environment
- `VGDLEnv.py` - Main game environment wrapper providing OpenAI Gym-like interface
- `VGDLEnvAndres.py` - Alternative environment implementation with additional features
- `vgdl/` - Complete VGDL (Video Game Description Language) framework for game logic
- `all_games/` - Game definitions and level files in VGDL format

### RL Models and Training
- `player.py` - Main RL agent implementation with PyTorch-based neural networks
- `rl_models.py` - Neural network architectures and replay memory implementations
- `runDDQN.py` - Training script for Double Deep Q-Network agents
- `utils.py` - Utility functions for game loading and data processing

### Human Data Analysis
- `scripts/replay_human_from_db.py` - Replay human gameplay from MongoDB database
- `scripts/visualize_fmri_bids.py` - fMRI data visualization tools
- `scripts/make_fmri_mp4.py` - Create MP4 videos from fMRI and gameplay data
- `scripts/plot_subject_metrics.py` - Behavioral data visualization

## Common Development Commands

### Environment Setup
```bash
# For Python 3 (recommended)
pip install -r requirements-py3.txt

# For Python 2.7 (legacy)
pip install -r requirements.txt

# For Dopamine integration
pip install absl-py atari-py gin-config gym opencv-python tensorflow-gpu
```

### Training Models
```bash
# Train DDQN on a specific game
python runDDQN.py -game_name aliens

# Train with custom parameters
python runDDQN.py -game_name sokoban -num_episodes 10000 -lr 0.001

# Run in test mode
python runDDQN.py -game_name aliens -test_mode 1 -model_weight_path path/to/weights
```

### Dopamine Integration
```bash
# Run Dopamine with VGDL games
python -um dopamine.discrete_domains.train \
  --base_dir=./tmp/dopamine/aliens \
  --gin_files='dopamine/agents/rainbow/configs/rainbow_aaaiAndres.gin' \
  --gin_bindings='create_atari_environment.game_name="VGDL_aliens"'
```

### Human Data Replay
```bash
# Replay human gameplay from MongoDB
python scripts/replay_human_from_db.py \
  --mongo-uri mongodb://localhost:27017 --db vgfmri \
  --dataset-root ~/Downloads/ds004323-download \
  --game-name vgfmri3_sokoban --level-idx 0 --limit 1

# Visualize with delay
python scripts/replay_human_from_db.py --visualize --delay-ms 80 [other args]

# Export to MP4
python scripts/replay_human_from_db.py --mp4-out output.mp4 --delay-ms 120 [other args]
```

## Data Structure

### MongoDB Collections (for fMRI dataset)
- `games` - Game definitions mapping names to VGDL text
- `plays` - Human gameplay episodes with action traces
- `subjects` - Subject registry
- `runs` - Scanner/game session metadata
- `regressors` - EMPA model regressors
- `dqn_regressors_25M` - DDQN model regressors

### Game Files
Games are defined in VGDL format in `all_games/` directory. Each game has:
- Main game file: `gamename.txt`
- Level files: `gamename_lvl0.txt`, `gamename_lvl1.txt`, etc.

### Key Configuration Options (runDDQN.py)
- `game_name` - Name of game to train on (matches file in all_games/)
- `num_episodes` - Number of training episodes (default: 20000)
- `batch_size` - Mini-batch size (default: 32)
- `lr` - Learning rate (default: 0.00025)
- `model_name` - Model architecture to use (default: 'DQN')
- `level_switch` - How to progress through levels ('sequential' or other)
- `random_seed` - Random seed for reproducibility (default: 7)

## Development Notes

### Python Version Support
- Primary development uses Python 3 with `requirements-py3.txt`
- Legacy Python 2.7 support available via `requirements.txt`
- PyBrain dependency removed in Python 3 version

### GPU Support
- Models automatically detect and use CUDA when available
- Set `-cuda 1` flag in runDDQN.py for GPU training

### Game Environment Interface
The VGDLEnv provides standard RL interface:
- `step(action)` - Execute action, return observation, reward, done, info
- `reset()` - Reset environment to initial state
- `set_level(level)` - Change to specific level
- `render()` - Get visual representation of current state

### Human Data Integration
For working with fMRI dataset (ds004323):
1. Restore MongoDB from `behavior/dump.tar.gz`
2. Use scripts in `scripts/` directory for analysis
3. Game definitions stored both in files and MongoDB `games` collection