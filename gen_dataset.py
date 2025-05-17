#!/usr/bin/env python
import math, multiprocessing as mp, pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm
MAX = 10_000_000            # overridable by CLI

def smallest_factor(n: int) -> int:
    if n % 2 == 0: return 2
    r = int(math.isqrt(n))
    f = 3
    while f <= r and n % f: f += 2
    return f if f <= r else n

def factorize(n: int) -> list[int]:
    factors = []
    while n > 1:
        f = smallest_factor(n)
        factors.append(f)
        n //= f
    return factors

def worker(chunk):
    records = []
    for n in chunk:
        records.append({"n": n, "fact": "×".join(map(str, factorize(n)))})
    return records

def main(max_n=MAX, workers=8):
    pool = mp.Pool(workers)
    chunks = [range(i, min(i+10000, max_n+1)) for i in range(2, max_n+1, 10000)]
    rows = []
    for recs in tqdm(pool.imap_unordered(worker, chunks), total=len(chunks)):
        rows.extend(recs)
    tbl = pa.Table.from_pylist(rows)
    pq.write_table(tbl, f"data/pf_{max_n}.parquet", compression="zstd")

if __name__ == "__main__":
    import argparse, os
    os.makedirs("data", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_n", type=int, default=MAX)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    main(**vars(ap.parse_args()))

