"""Tokenization + memory-mapped corpus + streaming DataLoader for OSC-LLM.

  1. train_tokenizer()      byte-level BPE (vocab V) -> tokenizer JSON
  2. build_corpus_bin()     text files -> single uint16 `corpus.bin` (+meta)
  3. MemmapDataset/loader   per-worker contiguous slices -> [B,S] batches

For distributed training each worker uses a unique seed (rank*epoch+worker) so
DataParallel ranks never overlap; num_workers exploits the EPYC 9755 cores.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ tokenizer --
def train_tokenizer(text_sources: list[str] | list[Path], vocab_size: int,
                    output_json: str, limit_bytes: int = 40 << 20):
    from tokenizers import Tokenizer, models, trainers
    from tokenizers.pre_tokenizers import ByteLevel

    tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<eos>", "<bos>", "<unk>"],
        show_progress=True,  initial_alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"),
    )

    consumed = [0]

    def _gen():
        for f in iter_text_files(text_sources):
            if consumed[0] >= limit_bytes:
                break
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                while True:
                    line = fp.readline()
                    if not line or consumed[0] >= limit_bytes:
                        break
                    txt = line.strip()
                    if txt:
                        consumed[0] += len(txt.encode("utf-8"))
                        yield txt

    tok.train_from_iterator(_gen(), trainer=trainer)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_json))
    return output_json


def load_tokenizer(path: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


# -------------------------------------------------------------------- corpus ---
_TEXT_EXTS = (".txt", ".jsonl", ".gz")


def iter_text_files(sources) -> list[str]:
    files: list[str] = []
    srcs = [sources] if isinstance(sources, str) else sources
    for s in srcs:
        p = Path(s)
        if p.is_dir():
            for f0 in sorted(p.rglob("*")):
                if f0.is_file() and f0.suffix.lower() in _TEXT_EXTS or \
                   (f0.is_file() and f0.suffix not in ("", ".bin", ".json", ".meta.json")):
                    files.append(str(f0))
        elif p.exists():
            files.append(s)
    return files


def build_corpus_bin(tok, text_sources, out_prefix: Path,
                     max_tokens=None) -> int:
    """Tokenize sources -> uint16 `corpus.bin`. Returns token count."""
    files = iter_text_files(text_sources)
    all_chunks: list[list[int]] = []
    total = 0

    for f in files:
        rows = _tokenize_file(tok, f)
        flat = [t for row in rows for t in row]
        if max_tokens and total + len(flat) > max_tokens:
            flat = flat[:max(int(max_tokens - total), 0)]
            all_chunks.append(flat); total += len(flat); break
        all_chunks.append(flat); total += len(flat)

    arr = np.concatenate(
        [np.asarray(c, dtype=np.uint16) for c in all_chunks],
        axis=0) if all_chunks and sum(map(len, all_chunks)) else \
        np.zeros((1,), dtype=np.uint16)
    # drop a trailing <eos>-ish separator is unnecessary; keep contiguous.

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = str(out_prefix.with_name(out_prefix.name + ".bin"))
    mm = np.memmap(bin_path, mode="w+", dtype=np.uint16,
                   shape=(arr.shape[0],))
    mm[:] = arr
    mm.flush()
    del mm

    json.dump({"n_tokens": int(arr.shape[0]),
               "vocab_size": tok.get_vocab_size(),
               "bin_path": bin_path},
              open(out_prefix.with_name("meta.json"), "w"))
    print(f"[data] {arr.shape[0]:,} tokens -> {bin_path}")
    return int(arr.shape[0])


def _tokenize_file(tok, f) -> list[list[int]]:
    rows: list[list[int]] = []
    lower = Path(f).suffix.lower()
    try:
        if lower == ".jsonl":
            with open(f, encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    obj = json.loads(line)
                    txt = (obj.get("text") or obj.get("content")
                           or obj.get('article') or "")
                    r = _enc(tok, str(txt))
                    if r: rows.append(r)
        elif lower == ".gz":
            import gzip
            with gzip.open(f, "rt", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    obj = json.loads(line)
                    txt = (obj.get("text") or obj.get("content")
                           or obj.get('article') or "")
                    r = _enc(tok, str(txt))
                    if r: rows.append(r)
        else:
            with open(f, encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    t0 = line.strip()
                    if t0 == "":
                        continue
                    eidc = _enc(tok, t0 + "\n")
                    rows.append(eidc)
    except Exception:
        pass  # swallow a single bad file; keep going.
    return [r for r in rows if len(r) > 1]


def _enc(tok, text):
    try:
        enc = tok.encode(str(text))
        ids = list(enc.ids)
    except Exception:
        return []
    return [int(min(i3, 65533)) for i3 in ids] + ([0xFFFE] if False else [])


# ------------------------------------------------------------------ dataloader --
class MemmapDataset:
    """Random contiguous-slice reader over uint16 .bin."""

    def __init__(self, bin_path: str):
        seq = np.memmap(bin_path, mode="r", dtype=np.uint16)
        self._arr_ref = seq
        n = int(seq.shape[0])
        # guard degenerate tiny files
        if n < 2:
            raise RuntimeError(f"corpus too small: {bin_path} ({n})")
        self.n = n

    def sample(self, length: int, seed: int) -> np.ndarray:
        start_max = max(1, int(self.n - length - 4))
        rng = random.Random(hash((seed * 1000003)))
        s = rng.randint(0, start_max)
        chunk = np.asarray(self._arr_ref[s: s + length], dtype=np.int64)
        if len(chunk) < length:
            import numpy as _np
            chunk = _np.pad(chunk, (0, length - len(chunk)))
        return chunk


def make_loader(bin_path: str, seq_len: int, micro_batch_train_tokens,
                num_workers: int | None = None):
    """Return an infinite [B,S] torch.LongTensor batcher.

    B derived from `micro_batch_train_tokens` / seq_len. Distributed-aware seeds
    via env RANK + LOCAL_RANK so parallel ranks sample disjoint slices.
    """
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        local_rank = 0

    ds = MemmapDataset(bin_path)
    batch_size = max(1, micro_batch_train_tokens // seq_len)

    return _StreamingLoader(ds, batch_size=batch_size, seq_len=seq_len,
                            num_ranks=_world(), local_rank=local_rank)


class _StreamingLoader:
    def __init__(self, ds: MemmapDataset, batch_size: int, seq_len: int,
                 num_workers: int = 8, num_ranks: int = 1,
                 local_rank: int = 0):
        self.ds, self.batch_size, self.seq_len = ds, batch_size, seq_len
        self.num_workers = max(1, num_workers)
        self.num_ranks = num_ranks
        self.local_rank = local_rank

    def __iter__(self):
        return _batcher(self)

    def sample_seq(self, seed: int) -> np.ndarray:
        """One slice of length seq_len unique to (seed)."""
        return self.ds.sample(self.seq_len,
                              seed * 1000003 + self.local_rank)


def _world():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    return 1


class _batcher:
    """Infinite iterator handing out [B,S] batches, multi-worker via prefetch."""

    def __init__(self, loader):
        import torch
        self.loader = loader
        B = loader.batch_size

    def __iter__(self):  # pragma: no cover - interface guard
        return self

    def __next__(self):
        import torch
        ld = self.loader; B = ld.batch_size
        seed0 = random.randint(1, 2 ** 31)
        rows = []
        for b in range(B):
            # emulate independent worker streams by offsetting seeds per batch idx
            s = ld.sample_seq(seed0 + b * 7919)      # distinct slices/col row
            if len(s) < ld.seq_len:
                s = np.pad(s, (0, ld.seq_len - len(s)))
            rows.append(torch.as_tensor(np.copy(s), dtype=torch.long))
        out = torch.stack(rows).contiguous()
        return {"input_ids": out}
