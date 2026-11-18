"""Đối chứng cắt tỉa và pattern trên cùng bộ vị trí hợp lệ."""

import argparse
from pathlib import Path

from agents.minimax_agent import MinimaxAgent
from agents.rule_based_agent import RuleBasedAgent
from config.search_budget import SearchBudget
from core.game_engine import CaroGame
from evaluation.tournament import fingerprint, write_json

POSITIONS = {
    "mo_dau": [(5, 5), (5, 6)],
    "trung_cuoc": [(5, 5), (5, 6), (6, 5), (4, 4), (6, 6), (4, 5)],
    "x_thang_ngay": [(0, 0), (10, 0), (0, 1), (10, 2), (0, 2), (10, 4), (0, 3), (10, 6)],
    "o_phai_chan": [(0, 0), (10, 0), (0, 1), (10, 2), (0, 2), (10, 4), (0, 3)],
    "ba_mo": [(5, 4), (10, 0), (5, 5), (10, 2), (5, 6)],
    "tao_bon_mo": [(5, 4), (10, 0), (5, 5), (10, 2), (5, 6), (10, 4)],
}

EXPECTED = {"x_thang_ngay": {(0, 4)}, "o_phai_chan": {(0, 4)},
            "ba_mo": {(5, 3), (5, 7)}, "tao_bon_mo": {(5, 3), (5, 7)}}


def run(output: Path) -> dict:
    results = []
    for name, moves in POSITIONS.items():
        game = CaroGame()
        for move in moves:
            game.step(move)
        for depth in (1, 2):
            comparison = []
            for pruning in (False, True):
                agent = MinimaxAgent(budget=SearchBudget(depth=depth), use_alpha_beta=pruning)
                action = agent.choose_action(game.board, game.current_player)
                comparison.append({"alpha_beta": pruning, "action": action, "stats": agent.stats.copy()})
            if comparison[0]["stats"]["root_value"] != comparison[1]["stats"]["root_value"]:
                raise ValueError(f"Giá trị gốc khác nhau khi bật cắt tỉa ở {name}, độ sâu {depth}")
            results.append({"position": name, "experiment": "alpha_beta", "depth": depth,
                            "moves": moves, "runs": comparison})
        comparison = []
        for patterns in (False, True):
            agent = RuleBasedAgent(use_patterns=patterns)
            action = agent.choose_action(game.board, game.current_player)
            comparison.append({"patterns": patterns, "action": action, "stats": agent.stats.copy(),
                               "tactical_correct": action in EXPECTED[name] if name in EXPECTED else None})
        results.append({"position": name, "experiment": "patterns", "moves": moves, "runs": comparison})
    report = {"source_and_checkpoint_sha256": fingerprint(), "positions": len(POSITIONS),
              "results": results,
              "scope": "Đối chứng trên bộ vị trí; thời gian và thay đổi nước đi được báo riêng, chưa suy ra tỷ lệ thắng."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(f"Đã lưu hai đối chứng trên {len(POSITIONS)} vị trí vào {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy đối chứng cắt tỉa và nhận diện pattern.")
    parser.add_argument("--output", type=Path, default=Path("results/statistics/ablations.json"))
    run(parser.parse_args().output)
