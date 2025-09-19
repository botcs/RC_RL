# ds004323 dataset overview (Csabi)

This note summarizes the folder hierarchy, where key data live, and how the MongoDB tables (collections) relate to the files so you can quickly navigate, replay, and analyze.

## Top‑level layout

- `README`, `CHANGES`, `dataset_description.json`: Dataset metadata (OpenNeuro/BIDS style).
- `games/`: VGDL game definitions and level files as plain text.
  - Files follow `vgfmri3_<game>.txt` and per‑level files `vgfmri3_<game>_lvl<N>.txt` (0‑based).
  - Example: `vgfmri3_sokoban.txt`, `vgfmri3_sokoban_lvl0.txt`, …
- `behavior/`: MongoDB dump of behavioral and modeling data.
  - Files: `dump.tar.gz` (and/or split `dump.tar.gz.parta*`), plus a `README` with restore instructions.
- `code/`: Reference code bundles (`DDQN.tar.gz`, `EMPA.tar.gz`, `analysis.tar.gz`). Not needed just to replay.
- `sub-01/`, `sub-02/`, …: Subject folders (BIDS). If present, contain imaging/behavioral files for each participant. (Not required for replays.)

## MongoDB contents (after restore)

Behavioral and derived data are organized into collections. The most useful ones for replays and analysis:

- `games`: Mapping from `name` → game/level text used in the study.
  - Fields: `name` (e.g., `vgfmri3_sokoban`), `descs` (list of game strings), `levels` (list of level strings).
- `subjects`: Subject registry. Key field: `subj_id` (integer, no PII).
- `runs`: Scanner/game runs. Key field: `run_id` (integer; 0 can be the practice run).
- `plays`: Episodes of actual gameplay with full keypress/action trace and timing.
  - Typical fields: `_id`, `subj_id`, `run_id`, `game_name`, `start_time`, `actions` (array of `[label, timestamp]`).
- `regressors`: EMPA regressors derived from `plays`.
- `dqn_regressors_25M`: DDQN regressors derived from `plays`.
- `plays_post`: Visual confound regressors, derived from `plays` after post‑processing.
- `sim_results`: Results from generative play (EMPA/DDQN).

Relationships:

- One `games.name` maps to VGDL text in `games.descs` (game) and `games.levels` (levels). Level indices are 0‑based.
- A `plays` document references `game_name` (string), `subj_id` and `run_id` (integers) and embeds the action sequence.
- Regressor tables are derived from `plays` (usually keyed by `_id` or shared fields).

## Quick start: restore + explore

1) Combine parts and restore (one‑time):

```bash
cd ~/Downloads/ds004323-download/behavior
cat dump.tar.gz.parta* > dump.tar.gz   # if parts exist
tar -xzvf dump.tar.gz
mongorestore dump/
```

2) Connect and browse with `mongosh`:

```bash
mongosh
show dbs
use <your_db_name>   # e.g., vgfmri
show collections

db.plays.findOne()
db.plays.find({game_name: "vgfmri3_sokoban", actions: {$ne: []}})

db.games.findOne({name: "vgfmri3_sokoban"})
```

3) Helpful queries:

```javascript
// First 5 non‑empty plays for a game
db.plays.find({game_name: "vgfmri3_sokoban", actions: {$ne: []}}, {subj_id:1, run_id:1, start_time:1})
         .sort({start_time:1}).limit(5)

// Distinct values
db.plays.distinct('game_name')
db.plays.distinct('subj_id')

// Longest episodes by actions length
db.plays.aggregate([
  {$match: {game_name: "vgfmri3_sokoban"}},
  {$project: {subj_id:1, run_id:1, actions_len: {$size: "$actions"}}},
  {$sort: {actions_len: -1}},
  {$limit: 5}
])

// Look up stored game/level text in DB
const g = db.games.findOne({name: "vgfmri3_sokoban"});
print(g.descs[0]);
print(g.levels[0]);
```

## Replaying episodes

You have two Python scripts in your repo (RC_RL):

- File streaming (no DB needed): `scripts/replay_human.py`
  - Reads `behavior/dump.tar.gz` directly, then uses `games/` files on disk.
  - Examples:
    - `--offset 3 --limit 1` to replay just the 4th matching play.
    - `--gif-out out.gif` or `--mp4-out out.mp4` to export media (install `imageio-ffmpeg` or `ffmpeg` for MP4).

- Live DB: `scripts/replay_human_from_db.py`
  - Connects to MongoDB (collections restored by `mongorestore`).
  - Can load game/level either from disk (`games/`) or from `db.games`.
  - Examples:

```bash
# First matching play from DB using disk game files
python scripts/replay_human_from_db.py \
  --mongo-uri mongodb://localhost:27017 --db <your_db_name> \
  --dataset-root ~/Downloads/ds004323-download \
  --game-name vgfmri3_sokoban --level-idx 0 --limit 1 --mp4-out out.mp4

# Same, but use game/level stored in MongoDB
python scripts/replay_human_from_db.py \
  --mongo-uri mongodb://localhost:27017 --db <your_db_name> \
  --games-source db --game-name vgfmri3_sokoban --limit 1 --gif-out out.gif
```

## Notes & tips

- MongoDB running: Either as a background service (e.g., `brew services start mongodb-community`) or on‑demand (manual `mongod` or Docker). Scripts need it available at runtime.
- Indexes (optional but useful):
  - `db.plays.createIndex({game_name:1, start_time:1})`
  - `db.plays.createIndex({subj_id:1, run_id:1, start_time:1})`
- Media export:
  - For MP4: install Python plugin `imageio-ffmpeg` or system `ffmpeg` for best results.
- Levels: If a `plays` doc has `level_idx`, prefer it; otherwise, pass `--level-idx`.
