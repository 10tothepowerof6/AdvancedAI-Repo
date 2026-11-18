"""Tổng hợp kết quả từ log và ước lượng bất định theo cặp ván đổi bên."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from config.search_budget import SearchBudget
from evaluation.metrics import MatchRecord, summarize_latencies, summarize_results
from evaluation.tournament import schedule, validate_game, write_json


def summarize(directory: Path) -> dict:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    budget = SearchBudget(**config["budget"])
    scheduled = schedule(config["agents"], config["games_per_pair"])
    saved_games = []
    for entry in scheduled:
        path = directory / f"{entry['id']}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            validate_game(saved, entry, budget)
            saved_games.append(saved)
    pairs = {}
    per_agent = {}
    for agent in config["agents"]:
        records = [MatchRecord(**saved["result"]) for saved in saved_games]
        logs = [log for saved in saved_games for log in saved["moves"]
                if (saved["game"]["agent_x"] if log["player"] == 1 else saved["game"]["agent_o"]) == agent
                and log["status"] == "move"]
        stats = summarize_results(records, agent)
        stats["latency"] = summarize_latencies(log["elapsed_s"] for log in logs)
        stats["nodes_visited"] = sum(log["stats"].get("nodes_visited", 0) for log in logs)
        stats["simulations"] = sum(log["stats"].get("simulations", 0) for log in logs)
        stats["by_side"] = {side: summarize_results(
            (record for record in records if getattr(record, field) == agent), agent)
            for side, field in (("X", "agent_x"), ("O", "agent_o"))}
        per_agent[agent] = stats

    grouped = defaultdict(list)
    for saved in saved_games:
        key = tuple(sorted((saved["game"]["agent_x"], saved["game"]["agent_o"])))
        grouped[key].append(saved)
    for (first, second), group in grouped.items():
        seed_scores = defaultdict(list)
        for saved in group:
            record = MatchRecord(**saved["result"])
            outcome = record.winner * (1 if record.agent_x == first else -1)
            seed_scores[saved["game"]["seed"]].append((outcome + 1) / 2)
        complete = [np.mean(scores) for scores in seed_scores.values() if len(scores) == 2]
        interval = None
        if len(complete) >= 2 and len(group) == 2 * len(complete):
            rng = np.random.default_rng(42)
            estimates = rng.choice(complete, size=(2000, len(complete))).mean(axis=1)
            interval = np.quantile(estimates, (0.025, 0.975)).tolist()
        pairs[f"{first}/{second}"] = {
            "agent": first, "opponent": second,
            **summarize_results((MatchRecord(**saved["result"]) for saved in group), first),
            "score_ci95_paired_bootstrap": interval, "complete_seed_pairs": len(complete)}
    return {"config": config, "scheduled_games": len(scheduled), "completed_games": len(saved_games),
            "complete": len(saved_games) == len(scheduled),
            "wall_seconds": sum(saved["wall_seconds"] for saved in saved_games),
            "agents": per_agent, "pairs": pairs,
            "confidence_method": "bootstrap_2000_by_seed_pairs_seed42"}


def main():
    parser = argparse.ArgumentParser(description="Tạo báo cáo từ log giải đấu.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"Đã tổng hợp {report['completed_games']}/{report['scheduled_games']} ván vào {args.output}")


if __name__ == "__main__":
    main()
