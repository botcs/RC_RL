# VGDL Directory Overview

This folder contains an implementation of VGDL (Video Game Description Language) and a research/experimentation stack around planning, Reinforcement learning, and theory induction in grid‑based games. It includes the core game engine, ontology (sprites, physics, interactions, terminations), RL environment wrappers, multiple planners (MCTS, width‑based, A*), Q‑learning and DP baselines, induction utilities for learning game rules, and supporting tools for visualization and experimentation.

Use this README as a map: start from the high‑level architecture, then drill down into the module listings to find concrete entry points and functionality.


High‑Level Architecture

- Core Engine
  - Parses VGDL text into a playable game; runs the game loop; handles sprites, collisions, rendering, and terminations.
  - Files: `core.py`, `ontology.py`, `stateobs.py`, `stateobsnonstatic.py`, `interfaces.py`, `__init__.py`, legacy variants in `core2.py`, `core_old.py`, `core_comments.py`.

- RL Environments
  - Wrap games for RL agents, providing sensors, action stepping, and termination signals. Both static‑other and fully non‑static variants.
  - Files: `rlenvironment.py`, `rlenvironmentnonstatic.py`.

- Planning and Search
  - Online planners, exploration heuristics, subgoal selection, and parallel orchestration. Includes MCTS, width‑based planning (IW/2BFS family), and A*.
  - Files: `mcts*.py`, `WBP*.py`, `aStar.py`, `planner.py`, `metaplanner.py`, `parallel_planning.py`, `nonparallel_planning.py`.

- Learning (RL/DP)
  - Q‑learning agent, MDP conversion utilities, and DP baselines (value iteration, RTDP).
  - Files: `qlearner*.py`, `mdpmap.py`, `value_iteration.py`, `rtdp.py`, `ai_algorithms.py`.

- Theory Induction and Priors
  - Induction of sprite classes and interaction rules from traces; theory templates; taxonomy and similarity helpers.
  - Files: `theory_template.py`, `class_theory_template.py`, `sprite_induction.py`, `induction.py`, `taxonomy.py`, `similarity.py`, `sampleVGDLString.py`, `outline.py`.

- Agents and Experiment Drivers
  - Agent orchestration for curricula, hyperparameter sweeps, and game loaders.
  - Files: `main_agent.py`, `agent*.py`, `load_games.py`, `hyperparameter_optimization.py`, loops in `bigloop*.py`.

- Utilities and Tools
  - Parsing utilities, geometry helpers, colors, plotting, keyboard helpers, video export, and YouTube upload wrapper.
  - Files: `tools.py`, `util.py`, `colors.py`, `plotting.py`, `keypress.py`, `make_videos.py`, `youtube.py`.

- Assets, Tests, and Notebooks
  - Ontology diagrams, tests, notebooks, and assorted scratch/benchmark artifacts.
  - Files: `VGDL_ontology.*`, `test*.py`, `notebooks/*`, `benchmarking`, `scraps.py`, `untitled`, `dump.rdb`.


Core Engine

- `core.py`: Main engine. `VGDLParser` parses VGDL text (SpriteSet, InteractionSet, LevelMapping, TerminationSet, ConditionalSet) into a `BasicGame`. Handles sprite construction, collision effect ordering, event handling, rendering, playback, level building (from map or positions), and termination checks.
- `ontology.py`: VGDL ontology. Defines physics (grid/continuous), directions, and the sprite class hierarchy (e.g., `Immovable`, `RandomNPC`, `Chaser`, `Missile`, `Avatar` variants), interactions/effects, resource mechanics, and termination classes (`Timeout`, `SpriteCounter`, `MultiSpriteCounter`, `NoveltyTermination`). Also includes helper constants and movement/distance utilities.
- `stateobs.py`: State/observation handler for games with static non‑avatar sprites. Encodes avatar position and optional orientation; builds local/global observations and presence vectors; enforces assumptions (grid physics, single avatar type).
- `stateobsnonstatic.py`: State/observation handler supporting moving/dynamic non‑avatar sprites. Tracks per‑type locations, handles kill lists, and returns consistent observations when sprites move or die.
- `interfaces.py`: PyBrain‑style wrappers (`GameEnvironment`, `GameTask`) around the engine for controlled rollouts and simple episodic tasks (legacy support).
- `core2.py`, `core_old.py`: Older/legacy versions of the core engine kept for reference.
- `core_comments.py`: Commented/reference variant of core routines.
- `__init__.py`: Package helper that aliases legacy absolute imports (e.g., `import ontology`) to `vgdl.ontology`, etc., to support both package and flat imports.


RL Environments

- `rlenvironment.py`: RL wrapper for games with static non‑avatar sprites. Supports local (neighborhood) or global grid observations, action stepping, and terminal rewards (`+1/-1/0`).
- `rlenvironmentnonstatic.py`: RL wrapper for general games with moving sprites and richer dynamics. Provides:
  - Global or local observations with encoded object‑type bits.
  - Keyboard‑style action input mapping and action stepping for all sprites.
  - Game construction helpers: `createMindEnv`, `createRLInputGame`, `createRLInputGameFromStrings`, plus canned `createRL*` for example games.
  - Display helpers (`show`, `show_binary`) and symbol dictionary for compact grid printing.
  - Extended termination handling (ordered checks including novelty termination).


Planning and Search

- `planner.py`: Base planning utilities. Scans the domain to compute feasible movement per cell (action/neighbor dictionaries), propagates pseudo‑rewards from goals, and provides heuristics (e.g., avoidance, reward shaping) used by planners.
- `aStar.py`: Subgoal‑oriented A* style planning on the grid derived from RL observations. Builds traversability maps and extracts subgoals along shortest paths.
- `mcts_clean.py`: Clean MCTS implementation for grid RL with reward shaping and neighborhood scanning; supports training and default policy rollouts; exposes best actions for playout.
- `mcts.py`, `mcts2.py`, `mcts_old.py`: Alternative/older MCTS variants kept for experimentation and regression.
- `mcts_pseudoreward_heuristic.py`, `mcts_pseudoreward_heuristic_b.py`: MCTS variants with pseudoreward heuristics (e.g., distance‑based shaping) and custom exploration/printing weights.
- `mcts_teleport.py`: MCTS variant with special handling for teleport mechanics.
- `WBP.py`, `WBP2.py`, `WBP3.py`, `WBP4.py`, `WBP5.py`, `WBP6.py`, `WBP8.py`: Width‑Based Planning family (IW(k), 2BFS variants). Encode novelty measures over atomized state features to guide breadth‑first exploration.
- `WBP_grid.py`: Width‑based planner tailored to continuous grid positions and orientation atoms; configurable horizon and limits, with optional wait/space actions.
- `WBP_class.py`, `WBP_one_hot.py`, `WBP_stable.py`: Specialized WBP variants experimenting with object class abstractions, one‑hot encodings, and stability/robustness tweaks.
- `metaplanner.py`: Orchestrates observation phases (sprite induction passes), event translation (IDs to colors), and top‑level plan/act loops over planners.
- `parallel_planning.py`, `nonparallel_planning.py`: Runners to schedule planner workloads in parallel or serial execution.
- `bigloop.py`, `bigloop2.py`: High‑level episodic loops combining theory induction, goal selection, planning to object goals, and optional playback/export.


Learning (RL/DP)

- `qlearner.py`: Q‑learning agent over grid observations. Epsilon‑greedy or greedy policies, reward shaping via pseudo‑rewards and avoidance scores, episodic training and annealing, and reporting/printouts.
- `qlearner_long.py`: Long‑horizon or extended Q‑learning variant with modified training/runtime settings.
- `mdpmap.py`: Converts a tractable VGDL game into an explicit MDP. Enumerates states via flood from the initial state, builds transition tensors `Ts[action][s][s']`, reward vector `R`, and optional feature map for state observations.
- `value_iteration.py`: Value‑iteration baseline for MDPs.
- `rtdp.py`: Real‑Time Dynamic Programming baseline.
- `ai_algorithms.py`: Adversarial/search utilities (minimax, alpha‑beta) adapted for simple games; mostly illustrative utilities.


Theory Induction and Priors

- `theory_template.py`: Core theory representation for rules and terminations. Defines:
  - TimeStep trace container; preconditions and rule classes.
  - Interaction rules (with precondition insertion), termination rules (e.g., `TimeoutRule`, `NoveltyRule`, `SpriteCounterRule`, `MultiSpriteCounterRule`).
  - Generation of a generic prior theory from a running game; writing theory and levels to text; helpers for mind‑environment creation.
- `class_theory_template.py`: Class/type‑level templates used by theory generation.
- `sprite_induction.py`: Online sprite type induction passed over game frames; maintains distributions, updates options, and infers sprite classes.
- `induction.py`: Runs theory induction from saved traces; supports DFS‑style hypothesis generation with trace rewriting (from object IDs to types) for compatibility.
- `taxonomy.py`: Taxonomy helpers used by theory construction and sprite classification.
- `similarity.py`: Utilities for computing similarity between (sprite/theory) structures.
- `sampleVGDLString.py`: Sample VGDL string templates for theory generation/testing.
- `outline.py`: Notes/outline for induction logic and class structure.
- `theoryTests.py`: Unit/regression tests for theory generation and round‑trip writing/reading.


Agents and Experiment Drivers

- `main_agent.py`: Orchestrates full agent runs across curricula. Manages per‑level theory generation, planning hyperparameters (including width‑based settings), annealing, recording, and video/image export. Entry point for large‑scale experiments.
- `agent.py`, `agents.py`: Agent wrappers (interactive user agent with keyboard input; policy‑driven agent using PyBrain policy iteration; legacy helpers).
- `agent2.py`, `agent_backup.py`, `agent_saved.py`: Alternative or legacy agent controllers kept for reference.
- `load_games.py`: Loads GVGAI game descriptions/levels (or local `all_games/*` descriptions), rewrites image refs to symbolic colors, constructs (game, level) pairs, and invokes `main_agent.Agent` across curricula. Accepts CLI args for hyperparameters and output video creation.
- `hyperparameter_optimization.py`: Hyperparameter sweep/optimization harness (e.g., via Hyperopt) for planner/theory parameters.


Utilities and Tools

- `tools.py`: Misc utilities: indented tree parser (`Node`, `indentTreeParser`) for VGDL text, vector/geometry helpers, event de‑dup (`oncePerStep`), rounded/square polygon generation, and GIF export (`makeGifVideo`).
- `util.py`: General helpers: softmax/normalize, Manhattan distance, bit‑factorization of cell encodings, object‑to‑symbol mapping, color lookup by object ID/type, random color generation, and CSV writing helpers.
- `colors.py`: Color constants and a `colorDict` mapping RGB tuples to symbolic names; includes a large palette and helpers to extend it.
- `plotting.py`: 2D feature plotting over grids and trajectory overlays using Matplotlib/Pylab.
- `keypress.py`: Curses‑based keyboard input helper for quick key detection demos.
- `make_videos.py`: Batch converts saved episode pickles into frame images and ffmpeg videos; organizes outputs by parameter set and game name.
- `youtube.py`: Thin wrapper around `external_libs.youtube_upload` for one‑shot YouTube uploads.


Additional Modules and Artifacts

- `olets_original.py`: Early/alternate MCTS/OLETS‑style agent scaffold with pseudoreward shaping and visit tracking (kept for comparison/experiments).
- `nonparallel_planning.py`, `parallel_planning.py`: Control scripts for planning workloads in serial vs. multiprocessing mode.
- `bigloop.py`, `bigloop2.py`: Long‑running plan/act loops that integrate induction, theory writing, goal selection, planning, and (optional) playback/visualization.
- `flexible_goals.py`, `goal_programming.py`: Alternative goal reasoning/programming utilities for directing planners.
- `outline.py`, `scraps.py`, `untitled`: Notes/scratch helpers retained for reference.
- `sampleVGDLString.py`: Templates for auto‑building VGDL games from theories.
- `VGDL_ontology.dot`, `VGDL_ontology.png`: Ontology diagram of sprite classes and relationships.
- `test.py`, `test_continuous.py`, `test_termination.py`: Small test drivers for engine physics and termination logic.
- `notebooks/debugging.ipynb`, `notebooks/test_theory_template.ipynb`: Jupyter notebooks for interactive debugging and theory template testing.
- `benchmarking`: Simple text artifact with timing comparisons for serial and parallel planning.
- `dump.rdb`: Serialized data artifact retained from experiments.
- `oles.java`: Reference/related Java code kept for historical comparison (not used by Python stack).


Typical Entry Points

- Parse and play a VGDL game: See `core.py` and use `VGDLParser.playGame(game_str, level_str, ...)`.
- Wrap a game for RL: Use `rlenvironmentnonstatic.py` helpers:
  - `createRLInputGame("examples.gridphysics.simpleGame4")` or
  - `createRLInputGameFromStrings(game_str, level_str)`
- Run a planner: Instantiate a WBP/MCTS planner using a created RLE (`RLEnvironmentNonStatic`) and call the appropriate training/plan methods (e.g., `Basic_MCTS.startTrainingPhase`, `getBestActionsForPlayout`).
- Full experiment driver: Use `main_agent.py` via `load_games.py` to run across GVGAI games/levels with configured hyperparameters.


Conventions and Notes

- Grid coordinates: Many planners use `(y, x)` when indexing numpy arrays; be mindful when converting from sprite rects `(left, top)` (pixels) to grid cells.
- Observations: Global observations encode avatar bit as `1`, other object types as bit‑shifts `2 << type_index`, and may be summed per cell; use helper `factorize` to decode.
- Legacy modules: Several `core*` and `mcts*` files are kept for comparison; prefer `core.py`, `rlenvironmentnonstatic.py`, `mcts_clean.py`, and `WBP_grid.py` for current workflows.
- External deps: The stack expects `pygame`, `numpy/scipy`, and (in some paths) `pybrain`, `matplotlib`, `termcolor`, and `ffmpeg` for video export.


File Index (by name)

- Agents: `agent.py`, `agent2.py`, `agents.py`, `agent_backup.py`, `agent_saved.py`, `main_agent.py`
- AI/Search: `ai.py`, `ai_algorithms.py`, `aStar.py`
- Core/Ontology: `core.py`, `core2.py`, `core_comments.py`, `core_old.py`, `ontology.py`, `ontology_comments.py`, `interfaces.py`, `stateobs.py`, `stateobsnonstatic.py`, `__init__.py`
- Environments: `rlenvironment.py`, `rlenvironmentnonstatic.py`
- Planning: `planner.py`, `metaplanner.py`, `parallel_planning.py`, `nonparallel_planning.py`, `mcts.py`, `mcts2.py`, `mcts_clean.py`, `mcts_old.py`, `mcts_pseudoreward_heuristic.py`, `mcts_pseudoreward_heuristic_b.py`, `mcts_teleport.py`, `WBP.py`, `WBP2.py`, `WBP3.py`, `WBP4.py`, `WBP5.py`, `WBP6.py`, `WBP8.py`, `WBP_class.py`, `WBP_grid.py`, `WBP_one_hot.py`, `WBP_stable.py`
- RL/DP: `qlearner.py`, `qlearner_long.py`, `mdpmap.py`, `value_iteration.py`, `rtdp.py`
- Induction/Rules: `theory_template.py`, `class_theory_template.py`, `sprite_induction.py`, `induction.py`, `taxonomy.py`, `similarity.py`, `sampleVGDLString.py`, `outline.py`, `theoryTests.py`
- Tools/Utils: `tools.py`, `util.py`, `colors.py`, `plotting.py`, `keypress.py`, `make_videos.py`, `youtube.py`
- Misc/Test: `test.py`, `test_continuous.py`, `test_termination.py`, `notebooks/*`, `VGDL_ontology.dot`, `VGDL_ontology.png`, `benchmarking`, `scraps.py`, `untitled`, `dump.rdb`, `oles.java`, `olets_original.py`


Questions or improvements

If you’d like this README to include runnable code snippets for a specific workflow in this repo (e.g., “solve Sokoban with WBP(k=2)”), let us know and we can add a focused quickstart section.
