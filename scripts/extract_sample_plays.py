#!/usr/bin/env python3
import argparse
import tarfile
import io
import sys

def find_member(t: tarfile.TarFile, suffix: str):
    for m in t:
        if m.name.endswith(suffix):
            return m
    return None

def main():
    p = argparse.ArgumentParser(description="Extract and print a few play docs from the MongoDB dump tar.gz without restoring")
    p.add_argument("--dataset-root", required=True, help="Path to ds004323 root (folder containing behavior/, games/, etc.)")
    p.add_argument("--max", type=int, default=3, help="Max number of play docs to show")
    args = p.parse_args()

    dump_tgz = f"{args.dataset_root}/behavior/dump.tar.gz"
    try:
        tf = tarfile.open(dump_tgz, mode="r:gz")
    except Exception as e:
        print(f"Failed to open {dump_tgz}: {e}")
        sys.exit(1)

    # Locate plays.bson (and maybe games.bson) inside the archive
    # Layout is usually dump/<dbname>/<collection>.bson
    tf.members = tf.getmembers()  # materialize to iterate twice safely
    plays_member = find_member(tf, "/plays.bson") or find_member(tf, "plays.bson")
    games_member = find_member(tf, "/games.bson") or find_member(tf, "games.bson")
    if not plays_member:
        print("Could not locate plays.bson in the dump archive.")
        for m in tf.members[:20]:
            print("-", m.name)
        sys.exit(2)

    # Lazy import bson from pymongo
    try:
        from bson import decode_file_iter
    except Exception:
        print("Missing bson (pymongo). Install pymongo and re-run.")
        sys.exit(3)

    # Read first few play docs
    f = tf.extractfile(plays_member)
    if f is None:
        print("Failed to extract plays.bson stream")
        sys.exit(4)

    count = 0
    for doc in decode_file_iter(f):
        count += 1
        print(f"\n=== Play #{count} ===")
        # Print helpful subset of fields if present
        keys_of_interest = [
            '_id', 'subj_id', 'run_id', 'game_name', 'game', 'level', 'level_idx',
            'keypresses', 'actions', 'steps', 'timestamps', 'start_time', 'duration'
        ]
        for k in keys_of_interest:
            if k in doc:
                v = doc[k]
                if isinstance(v, (list, tuple)):
                    print(f"{k}: len={len(v)} sample={v[:5]}")
                else:
                    print(f"{k}: {v}")
        # If none of the above matched, just print keys
        if not any(k in doc for k in keys_of_interest):
            print("keys:", sorted(list(doc.keys())))

        if count >= args.max:
            break

    # Optionally, also show one games doc
    if games_member:
        gf = tf.extractfile(games_member)
        if gf is not None:
            try:
                for i, gdoc in enumerate(decode_file_iter(gf)):
                    print("\n=== Game Doc Sample ===")
                    print("keys:", sorted(list(gdoc.keys())))
                    if 'name' in gdoc:
                        print('name:', gdoc['name'])
                    if 'game' in gdoc:
                        print('game snippet:', str(gdoc['game'])[:200].replace('\n', ' '))
                    if 'level' in gdoc:
                        print('level snippet:', str(gdoc['level'])[:200].replace('\n', ' '))
                    break
            except Exception:
                pass

if __name__ == "__main__":
    main()

