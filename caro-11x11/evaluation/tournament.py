"""Giải đấu đổi bên, lưu từng ván và tiếp tục từ kết quả đã kiểm tra."""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
from math import isfinite
from pathlib import Path
import platform
import sys
from time import perf_counter

from filelock import FileLock, Timeout

from config.search_budget import SearchBudget
from core.game_engine import CaroGame
from evaluation.match import AgentExecutionError, play_match
from evaluation.metrics import MatchRecord
from evaluation.registry import AGENT_NAMES, MAIN_AGENTS, AgentSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def schedule(agents: list[str], games_per_pair: int) -> list[dict]:
    if len(agents) < 2 or len(set(agents)) != len(agents):
        raise ValueError("Cần ít nhất hai tác tử khác nhau")
    if isinstance(games_per_pair, bool) or games_per_pair < 2 or games_per_pair % 2:
        raise ValueError("Số ván mỗi cặp phải chẵn và ít nhất bằng 2")
    for name in agents:
        AgentSpec(name)
    games = []
    for first, second in combinations(agents, 2):
        for seed in range(games_per_pair // 2):
            for side, (agent_x, agent_o) in enumerate(((first, second), (second, first))):
                games.append({"id": f"{first}_{second}_{seed:04d}_{side}", "seed": seed,
                              "agent_x": agent_x, "agent_o": agent_o})
    return games


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint() -> str:
    digest = hashlib.sha256()
    for directory in ("agents", "core", "config", "evaluation", "training"):
        for path in sorted((PROJECT_ROOT / directory).glob("*.py")):
            digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    digest.update((PROJECT_ROOT / "models/caro-policy-v1/policy.pt").read_bytes())
    return digest.hexdigest()


def validate_game(saved: dict, expected: dict, budget: SearchBudget) -> None:
    if saved["game"] != expected or saved["budget"] != asdict(budget):
        raise ValueError("Kết quả ván không khớp lịch hoặc ngân sách")
    record = MatchRecord(**saved["result"])
    if (record.agent_x, record.agent_o) != (expected["agent_x"], expected["agent_o"]):
        raise ValueError("Tên tác tử trong kết quả không khớp lịch")
    game = CaroGame()
    logs = saved["moves"]
    for index, log in enumerate(logs):
        expected_budget = asdict(budget)
        expected_budget.pop("mode")
        limit = budget.time_limit_s if budget.mode == "time" else 30.0
        if (log.get("mode") != budget.mode or log.get("budget") != expected_budget
                or log.get("limit_s") != limit):
            raise ValueError("Ngân sách trong log không khớp cấu hình ván")
        elapsed = log["elapsed_s"]
        if not isinstance(elapsed, (float, int)) or not isfinite(elapsed) or elapsed < 0:
            raise ValueError("Thời gian trong log không hợp lệ")
        if game.is_terminal():
            raise ValueError("Log tiếp tục sau khi ván đã kết thúc")
        if log["player"] != game.current_player or log["ply"] != game.move_count + 1:
            raise ValueError("Log không đúng thứ tự lượt đi")
        if log["status"] == "move":
            if elapsed > limit:
                raise ValueError("Nước được ghi hợp lệ nhưng vượt ngân sách")
            game.step(tuple(log["action"]))
        elif index != len(logs) - 1 or record.reason != log["status"] or record.winner != -game.current_player:
            raise ValueError("Log xử thua không khớp kết quả")
    if record.reason == "completed":
        if not game.is_terminal() or game.result() != record.winner:
            raise ValueError("Kết quả không khớp ván phát lại")
    elif not logs or logs[-1]["status"] != record.reason:
        raise ValueError("Ván xử thua thiếu log lỗi")


def run_tournament(output: Path, agents: list[str], *, games_per_pair: int,
                   budget: SearchBudget) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".run.lock", timeout=0):
        _run_locked(output, agents, games_per_pair=games_per_pair, budget=budget)


def _run_locked(output: Path, agents: list[str], *, games_per_pair: int,
                 budget: SearchBudget) -> None:
    games = schedule(agents, games_per_pair)
    identity = {"schema": 1, "agents": agents, "games_per_pair": games_per_pair,
                "budget": asdict(budget), "source_and_checkpoint_sha256": fingerprint(),
                "python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
                "libraries": {name: version(name) for name in ("numpy", "torch")},
                "rules": "freestyle_11x11_at_least_5", "watchdog_s": 30.0}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Thư mục có cấu hình khác. Hãy chọn thư mục mới")
    else:
        if any(path.name != ".run.lock" for path in output.iterdir()):
            raise ValueError("Thư mục kết quả có tệp nhưng thiếu cấu hình")
        write_json(config_path, identity)
    started = perf_counter()
    completed = 0
    for entry in games:
        destination = output / f"{entry['id']}.json"
        if destination.exists():
            validate_game(json.loads(destination.read_text(encoding="utf-8")), entry, budget)
            continue
        start = perf_counter()
        result, logs = play_match(AgentSpec(entry["agent_x"], entry["seed"]),
                                  AgentSpec(entry["agent_o"], entry["seed"]),
                                  agent_x=entry["agent_x"], agent_o=entry["agent_o"], budget=budget)
        saved = {"game": entry, "budget": asdict(budget), "result": asdict(result),
                 "moves": logs, "wall_seconds": perf_counter() - start}
        validate_game(saved, entry, budget)
        write_json(destination, saved)
        completed += 1
        print(f"Đã lưu {entry['id']} | thắng {result.winner} | {result.reason} | "
              f"{saved['wall_seconds']:.1f} giây", flush=True)
    print(f"Hoàn tất {len(games)} ván; chạy mới {completed} ván trong {perf_counter()-started:.1f} giây.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy hoặc tiếp tục giải đấu Caro.")
    parser.add_argument("--output", type=Path, required=True, help="Thư mục lưu từng ván")
    parser.add_argument("--agents", nargs="+", choices=tuple(AGENT_NAMES), default=list(MAIN_AGENTS))
    parser.add_argument("--games-per-pair", type=int, default=100)
    parser.add_argument("--mode", choices=("fixed", "time"), default="fixed")
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=2.0)
    args = parser.parse_args()
    try:
        run_tournament(args.output, args.agents, games_per_pair=args.games_per_pair,
                       budget=SearchBudget(args.mode, args.simulations, args.depth, args.time_limit))
    except Timeout:
        parser.exit(2, "Thư mục đang có một lượt chạy khác. Hãy chờ lượt đó kết thúc.\n")
    except (OSError, ValueError, AgentExecutionError) as error:
        parser.exit(2, f"Không thể tiếp tục giải đấu: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Đã dừng. Các ván đã lưu được giữ lại; dùng cùng lệnh để chạy tiếp.\n")


if __name__ == "__main__":
    main()
