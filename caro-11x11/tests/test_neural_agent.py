"""Kiểm thử tác tử sinh dữ liệu và toàn bộ luồng chuẩn bị mẫu huấn luyện."""

from collections import defaultdict
import hashlib
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from core.game_engine import CaroGame
from training.dataset import SPLITS, PolicyDataset, canonical_key, normalize_board, transform_sample
from training.generate_data import generate_dataset
from training.model import CaroPolicy, load_policy, mask_occupied, save_policy
from training.tactical_agent import TacticalAgent
from training.train import _run_epoch


def _position(moves):
    game = CaroGame()
    for move in moves:
        game.step(move)
    return game


@pytest.mark.parametrize(
    "moves,expected,player",
    [
        ([(0, 0), (10, 0), (0, 1), (10, 1), (0, 2), (10, 2), (0, 3), (10, 3)], (0, 4), 1),
        ([(10, 8), (0, 0), (10, 0), (1, 1), (10, 2), (2, 2), (10, 4), (3, 3), (10, 6)], (4, 4), -1),
        ([(0, 1), (0, 0), (0, 2), (0, 7), (0, 3), (10, 9), (0, 5), (10, 10),
          (0, 6), (10, 8)], (0, 4), 1),
    ],
    ids=["x_thang_truoc_khi_chan", "o_thang_cheo", "sau_quan_bi_chan_hai_dau"],
)
def test_teacher_takes_immediate_win(moves, expected, player):
    game = _position(moves)
    before = game.board.copy()
    action = TacticalAgent(42).choose_action(game.board, game.current_player)
    assert action == expected
    np.testing.assert_array_equal(game.board, before)
    game.step(action)
    assert game.result() == player


@pytest.mark.parametrize(
    "moves,expected",
    [
        ([(0, 0), (10, 1), (0, 1), (10, 3), (0, 2), (10, 5), (0, 3)], {(0, 4)}),
        ([(10, 1), (0, 0), (10, 3), (0, 1), (10, 5), (0, 2), (10, 7), (0, 3)], {(0, 4)}),
        ([(5, 3), (10, 1), (5, 4), (10, 3), (5, 5), (10, 5), (5, 6)], {(5, 2), (5, 7)}),
    ],
    ids=["o_chan_x", "x_chan_o", "chan_mot_trong_hai_de_doa"],
)
def test_teacher_blocks_immediate_threat(moves, expected):
    game = _position(moves)
    assert TacticalAgent(42).choose_action(game.board, game.current_player) in expected


def test_teacher_prefers_two_future_wins():
    game = _position([(5, 4), (10, 0), (5, 5), (10, 2), (5, 6), (10, 4)])
    assert TacticalAgent(42).choose_action(game.board, game.current_player) in {(5, 3), (5, 7)}


def test_normalize_for_both_players():
    game = _position([(5, 5), (5, 6)])
    own_x = normalize_board(game.board, 1)
    own_o = normalize_board(game.board, -1)
    assert (own_x[5, 5], own_x[5, 6]) == (1, -1)
    assert (own_o[5, 5], own_o[5, 6]) == (-1, 1)
    assert own_x.dtype == np.int8 and own_o.dtype == np.int8
    np.testing.assert_array_equal(own_x, -own_o)


def test_symmetries_move_board_and_label_together():
    game = _position([(2, 3), (4, 7)])
    board = normalize_board(game.board, game.current_player)
    marked = board.copy()
    marked[1, 2] = 9
    for symmetry in range(8):
        transformed, action = transform_sample(board, 13, symmetry)
        transformed_marker, _ = transform_sample(marked, 13, symmetry)
        assert transformed.ravel()[action] == 0
        assert transformed_marker.ravel()[action] == 9
        assert canonical_key(transformed) == canonical_key(board)
    _, rotated_action = transform_sample(board, 13, 1)
    assert divmod(rotated_action, 11) == (8, 1)


def test_generated_dataset_is_replayable_disjoint_and_deterministic(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = generate_dataset(first, games=24, seed=42)
    assert generate_dataset(second, games=24, seed=42) == manifest
    assert sum(manifest["outcomes"].values()) == 24
    for name, checksum in manifest["sha256"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == checksum

    expected = defaultdict(dict)
    seen_keys, seen_games = set(), set()
    for split in SPLITS:
        dataset = PolicyDataset(first, split)
        stats = manifest["splits"][split]
        assert len(dataset) == stats["samples"] > 0
        assert len(dataset) + stats["removed_duplicates"] == stats["samples_before_dedup"]
        game_ids = set(dataset.game_ids)
        assert not seen_games & game_ids
        seen_games.update(game_ids)
        for index, (board, action, game_id, ply) in enumerate(zip(
                dataset.boards, dataset.actions, dataset.game_ids, dataset.plies)):
            key = canonical_key(board)
            assert key not in seen_keys
            seen_keys.add(key)
            assert board.ravel()[action] == 0
            assert 0 <= action < 121
            channels, label = dataset[index]
            assert channels.shape == (2, 11, 11) and channels.dtype == np.float32
            np.testing.assert_array_equal(channels[0] - channels[1], board)
            assert label == action
            expected[game_id][ply] = (board, int(action), split)

    records = [json.loads(line) for line in (first / "games.jsonl").read_text().splitlines()]
    for record in records:
        game = CaroGame()
        for ply, move in enumerate(record["moves"], 1):
            if ply in expected[record["game_id"]]:
                board, action, split = expected[record["game_id"]].pop(ply)
                assert ply > record["opening_plies"]
                assert split == record["split"] and divmod(action, 11) == tuple(move)
                np.testing.assert_array_equal(board, game.board * game.current_player)
            game.step(tuple(move))
        assert game.is_terminal() and game.result() == record["result"]
    assert all(not positions for positions in expected.values())

    augmented = PolicyDataset(first, "train", augment=True)
    assert len(augmented) == 8 * manifest["splits"]["train"]["samples"]
    channels, action = augmented[1]
    board, expected_action = transform_sample(augmented.boards[0], int(augmented.actions[0]), 1)
    np.testing.assert_array_equal(channels[0] - channels[1], board)
    assert action == expected_action
    with pytest.raises(ValueError, match="tập train"):
        PolicyDataset(first, "test", augment=True)
    with pytest.raises(FileExistsError, match="đã tồn tại"):
        generate_dataset(first, games=24, seed=42)


def test_cnn_policy_masks_both_players_and_keeps_legal_logits():
    boards = torch.zeros((2, 2, 11, 11))
    boards[0, 0, 0, 1] = 1
    boards[0, 1, 1, 0] = 1
    boards[1, 1, 2, 3] = 1

    with torch.inference_mode():
        raw = CaroPolicy()(boards)
        masked = mask_occupied(raw, boards)
    assert raw.shape == (2, 121)
    assert torch.isfinite(raw).all()
    assert torch.isneginf(masked).nonzero().tolist() == [[0, 1], [0, 11], [1, 25]]
    torch.testing.assert_close(masked[1, :25], raw[1, :25])


def test_training_step_updates_weights_with_finite_loss():
    boards = torch.zeros((2, 2, 11, 11))
    boards[0, 0, 5, 5] = 1
    boards[1, 1, 5, 6] = 1
    labels = torch.tensor([59, 60])
    loader = DataLoader(TensorDataset(boards, labels), batch_size=2)
    model = CaroPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = _run_epoch(model, loader, torch.device("cpu"), optimizer)

    assert 0 < metrics["loss"] < float("inf")
    assert 0 <= metrics["top1"] <= metrics["top3"] <= 1
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_checkpoint_restores_same_logits(tmp_path):
    path = tmp_path / "policy.pt"
    model = CaroPolicy().eval()
    boards = torch.zeros((1, 2, 11, 11))
    boards[0, 1, 3, 5] = 1

    save_policy(path, model, epoch=2, dataset_sha256="fixture")
    restored = load_policy(path)

    assert not restored.training
    with torch.inference_mode():
        torch.testing.assert_close(restored(boards), model(boards), atol=0, rtol=0)
