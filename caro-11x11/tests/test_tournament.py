"""Kiểm thử lịch đấu, log, chạy tiếp và độ nhất quán thống kê."""

from dataclasses import asdict

import pytest

from config.search_budget import SearchBudget
from core.game_engine import CaroGame
from evaluation.analyze import summarize
from evaluation.metrics import MatchRecord
from evaluation.registry import AgentSpec
from evaluation.tournament import run_tournament, schedule, validate_game


def _draw_game(factory_x, factory_o, *, agent_x, agent_o, budget):
    from agents.random_agent import RandomAgent
    game = CaroGame()
    agent = RandomAgent(seed=3)
    logs = []
    while not game.is_terminal():
        player = game.current_player
        action = agent.choose_action(game.board, player)
        game.step(action)
        logs.append({"player": player, "ply": game.move_count, "status": "move", "action": action,
                     "elapsed_s": 0.01, "stats": {}, "mode": budget.mode,
                     "limit_s": budget.time_limit_s if budget.mode == "time" else 30.0,
                     "budget": {"simulations": budget.simulations, "depth": budget.depth,
                                "time_limit_s": budget.time_limit_s}})
    return MatchRecord(agent_x, agent_o, game.result()), logs


def test_schedule_balances_sides_and_seeds():
    games = schedule(["random", "neural", "mcts", "rule", "minimax"], 100)
    assert len(games) == 1000 and len({game["id"] for game in games}) == 1000
    for first, second in zip(games[::2], games[1::2]):
        assert first["seed"] == second["seed"]
        assert (first["agent_x"], first["agent_o"]) == (second["agent_o"], second["agent_x"])
    with pytest.raises(ValueError):
        schedule(["random", "neural"], 3)
    with pytest.raises(ValueError):
        schedule(["random", "random"], 2)


def test_resume_and_summary_match_saved_games(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.tournament.play_match", _draw_game)
    output = tmp_path / "run"
    budget = SearchBudget()
    run_tournament(output, ["random", "neural"], games_per_pair=4, budget=budget)
    saved_before = {path.name: path.read_bytes() for path in output.glob("*.json")}
    run_tournament(output, ["random", "neural"], games_per_pair=4, budget=budget)
    assert saved_before == {path.name: path.read_bytes() for path in output.glob("*.json")}
    summary = summarize(output)
    assert summary["complete"] and summary["completed_games"] == 4
    assert summary["agents"]["random"]["total"] == 4
    assert summary["agents"]["random"]["score"] == 0.5
    assert summary["agents"]["random"]["by_side"]["X"]["total"] == 2
    assert summary["pairs"]["neural/random"]["score_ci95_paired_bootstrap"] == [0.5, 0.5]
    with pytest.raises(ValueError, match="cấu hình khác"):
        run_tournament(output, ["random", "neural"], games_per_pair=6, budget=budget)


def test_timeout_validation_rejects_wrong_winner():
    budget = SearchBudget(mode="time")
    entry = schedule(["random", "neural"], 2)[0]
    saved = {"game": entry, "budget": asdict(budget),
             "result": asdict(MatchRecord("random", "neural", -1, "timeout")),
             "moves": [{"player": 1, "ply": 1, "status": "timeout", "elapsed_s": 2.01,
                        "limit_s": 2.0, "mode": "time", "budget": {"simulations": 1000,
                                                                   "depth": 2, "time_limit_s": 2.0}}]}
    validate_game(saved, entry, budget)
    saved["result"]["winner"] = 1
    with pytest.raises(ValueError, match="xử thua"):
        validate_game(saved, entry, budget)


def test_locked_run_and_corrupted_log_are_rejected(tmp_path, monkeypatch):
    import json
    from filelock import FileLock, Timeout
    monkeypatch.setattr("evaluation.tournament.play_match", _draw_game)
    output = tmp_path / "run"
    output.mkdir()
    with FileLock(output / ".run.lock"):
        with pytest.raises(Timeout):
            run_tournament(output, ["random", "neural"], games_per_pair=2, budget=SearchBudget())
    run_tournament(output, ["random", "neural"], games_per_pair=2, budget=SearchBudget())
    path = output / (schedule(["random", "neural"], 2)[0]["id"] + ".json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["moves"][0]["mode"] = "time"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="Ngân sách trong log"):
        run_tournament(output, ["random", "neural"], games_per_pair=2, budget=SearchBudget())


@pytest.mark.parametrize("kind", ("random", "rule", "minimax", "mcts", "neural"))
def test_factories_create_fresh_agents(kind):
    factory = AgentSpec(kind, 4)
    assert factory() is not factory()
