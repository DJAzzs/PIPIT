"""Download & sample Chinese+code corpora from ModelScope into dataset/.

Writes files under <root>/dataset for tokenizer training + data preparation.
Downloads are BOUNDED so we never pull SkyPile's full ~1TB scale accidentally.

    python scripts/download_data.py --datasets opencsg/chinese-fineweb-edu \
                modelscope/SkyPile-150B --target_gb 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _discover(api, repo_id: str) -> list[dict]:
    """Return leaf file entries (name/size/path/relative) via pagination."""
    out = {}
    for pg in range(1, 201):                      # page_size=100 hard cap on hub
        try:
            got = api.get_dataset_files(repo_id=repo_id,
                                        page_size=100, page_number=pg)
        except Exception as e:                    # some hubs error past end
            break
        if not got:
            break
        for ent in got:
            typ = ent.get("Type")
            full = str(ent.get("Path") or "")
            nm = ent.get("Name", os.path.basename(full))
            sz = int(ent.get("Size") or 0)
            # keep files (not `tree` dirs). Also filter out directory-like sizes.
            if typ == "blob" and "." in nm:
                rel = full.split("/", 1)[-1]
                out[full] = dict(size=sz, path=full,
                                 name=nm, relative=rel)
    return list(out.values())


def download_bounded(repo: str, out_dir: Path, target_bytes: int):
    from modelscope.hub.api import HubApi
    api = HubApi()
    try:
        leaves = _discover(api, repo)
    except Exception as e:
        print(f"[download] failed listing {repo}: {e}", file=sys.stderr)
        return None

    dl_dir = out_dir / repo.replace("/", "__")
    # pick modest files first; skip directories/oversize to avoid TB pulls.
    leaves.sort(key=lambda x: x["size"])
    chosen, acc = [], 0
    for f in leaves:
        if target_bytes - acc <= 0:
            break
        sz = f["size"]
        if 0 < sz <= (target_bytes - acc):
            chosen.append(f)
            acc += sz

    print(f"[download] {repo}: selected {len(chosen)} files ~{acc/1e9:.2f}GB")
    if not chosen:
        return dl_dir

    from modelscope import snapshot_download
    patterns = [c["relative"] for c in chosen[-64:]]
    try:
        out = snapshot_download(repo_id=repo, allow_patterns=list(set(patterns)),
                                local_dir=str(dl_dir))
        print(f"[download] -> {out}")
    except Exception as e:
        print(f"[download] snapshot error {repo}: {e}", file=sys.stderr)
    return dl_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["opencsg/chinese-fineweb-edu",
                             "modelscope/SkyPile-150B"])
    ap.add_argument("--target_gb", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    proj = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out) if args.out else proj / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    per = int(args.target_gb * 1024 ** 3 // max(len(args.datasets), 1))
    for ds in args.datasets:
        try:
            download_bounded(ds, out_dir, per)
        except Exception as e:
            print(f"[download] FAILED {ds}: {e}", file=sys.stderr)

    print("\n[download] Done. Inspect with:")
    print(f"   find '{out_dir}' -type f | head")


if __name__ == "__main__":
    main()
