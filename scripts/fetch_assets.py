#!/usr/bin/env python3
"""Fetch pinned upstream sources. No source assets are rewritten."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run(args, check=True)


def main():
    lock = json.loads((ROOT / "assets.lock.json").read_text())
    for name, entry in lock.items():
        destination = ROOT / "third_party" / name
        if destination.exists():
            actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
            if actual != entry["commit"]:
                raise RuntimeError(f"{destination}: expected {entry['commit']}, found {actual}. Existing checkout left untouched.")
            sparse = subprocess.run(["git", "-C", str(destination), "config", "--bool", "core.sparseCheckout"],
                                    capture_output=True, text=True)
            if sparse.stdout.strip() == "true":
                # Expand updated lock entries without removing extra paths the
                # user may have checked out independently.
                run("git", "-C", str(destination), "sparse-checkout", "add", *entry["sparse"])
            print(f"Verified {name}@{actual}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            run("git", "init", str(destination))
            run("git", "-C", str(destination), "remote", "add", "origin", entry["url"])
            run("git", "-C", str(destination), "sparse-checkout", "init", "--cone")
            run("git", "-C", str(destination), "sparse-checkout", "set", *entry["sparse"])
            run("git", "-C", str(destination), "fetch", "--depth", "1", "--filter=blob:none", "origin", entry["commit"])
            run("git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD")


if __name__ == "__main__":
    main()
