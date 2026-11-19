"""Run a prefix-only allocator on an immutable bank of precomputed losses."""
import argparse
import json
from pathlib import Path
import numpy as np
from dataset import CASES, CORPUS, load_losses
from playback import POLICIES, PrefixPlayback, run_policy


def choose_arm(observation):
    """A greedy starter: replace this function with your own allocation rule."""
    pools, arms = observation.losses.shape
    if observation.total_pulls < arms:
        return np.full(pools, observation.total_pulls, dtype=np.int32)
    return np.argmin(observation.losses, axis=1).astype(np.int32)


def play_greedy(loss, budget):
    env = PrefixPlayback(loss)
    observation = env.observation()
    pools = loss.shape[0]
    best = observation.losses.min(axis=1)
    values = np.empty((pools, budget + 1))
    values[:, 0] = best
    choices = np.empty((pools, budget), dtype=np.uint16)
    for t in range(budget):
        selected = choose_arm(observation)
        observation = env.pull(selected)
        best = np.minimum(best, observation.losses[np.arange(pools), selected])
        choices[:, t] = selected
        values[:, t + 1] = best
    return values, choices, observation.counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="near_saddle_gd")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--arms", type=int, default=32)
    parser.add_argument("--pools", type=int, help="Optional first N pools for a quick check")
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=514)
    parser.add_argument("--policy", default="greedy", choices=[
        "greedy", *POLICIES, "adaptive_matched", "adaptive_matched_floor"])
    parser.add_argument("--out", type=Path, help="Optional NPZ output of revealed outcomes and decisions")
    args = parser.parse_args()
    loss = load_losses(args.case, args.arms, args.corpus)
    if not args.arms <= args.budget <= loss.shape[2] - 1:
        parser.error("budget must include one warm-up pull per arm and not exceed the recorded horizon")
    if args.pools is not None:
        if not 1 <= args.pools <= len(loss):
            parser.error("pools must be between 1 and the number of available pools")
        loss = loss[:args.pools]
    if args.policy == "greedy":
        values, choices, counts = play_greedy(loss, args.budget)
    elif args.policy.startswith("adaptive"):
        if args.arms > 64:
            parser.error("the supplied probability-matching implementation supports at most 64 arms")
        from adaptive import run
        values, choices, counts, _, _, _ = run(
            loss, args.seed, "matched", .1 if args.policy.endswith("floor") else 0.,
            horizon=args.budget)
    else:
        values, choices, counts = run_policy(loss, args.policy, args.seed, horizon=args.budget)
    summary = dict(case=args.case, policy=args.policy, arms=args.arms, pools=len(loss),
                   budget=args.budget, seed=args.seed,
                   median_best_revealed_loss=float(np.median(values[:, -1])))
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, best_loss=values, choices=choices, final_counts=counts,
                            metadata_json=json.dumps(summary))


if __name__ == "__main__":
    main()
