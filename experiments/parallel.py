from multiprocessing import Pool

from experiments.runner import run_one
from train.data import load_dataset

_cache = {}


def _data(name, seed):
    if (name, seed) not in _cache:
        _cache.clear()
        _cache[(name, seed)] = load_dataset(name, seed)
    return _cache[(name, seed)]


def _job(job):
    job = dict(job)
    variant = job.pop("variant", job.get("criterion", ""))
    dataset = job.pop("dataset", "mnist")
    seed = job.pop("seed")
    res = run_one(_data(dataset, seed), seed, **job)
    row = res["row"]
    row["variant"] = variant
    return row


def run_jobs(jobs, workers=6):
    results = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap(_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {row['variant']} s={row['target_sparsity']} "
                  f"seed={row['seed']} val={row['val_acc']:.4f} test={row['test_acc']:.4f}", flush=True)
            results.append(row)
    return results
