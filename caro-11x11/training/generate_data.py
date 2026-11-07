"""Sinh bộ dữ liệu policy Caro từ các ván tự đấu có thể phát lại."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import numpy as np

from config.game_config import BOARD_SIZE, WIN_LENGTH
from core.game_engine import CaroGame
from core.game_utils import candidate_actions
from training.dataset import SPLITS, canonical_key, normalize_board
from training.tactical_agent import TacticalAgent

DEFAULT_OUTPUT = Path("training/data/caro11-v1")


@dataclass(frozen=True)
class Sample:
    board: np.ndarray
    action: int
    game_id: int
    ply: int


def _assign_splits(games: int, rng: np.random.Generator) -> dict[int, str]:
    held_out = max(1, games // 10)
    groups = np.split(rng.permutation(games), (games - 2 * held_out, games - held_out))
    return {int(game_id): split for split, group in zip(SPLITS, groups) for game_id in group}


def _play_game(game_id: int, seed: int, split: str, opening_plies: int) -> tuple[dict, list[Sample]]:
    opening_seed, teacher_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(opening_seed)
    agent = TacticalAgent(int(teacher_seed.generate_state(1)[0]))
    opening_count = int(rng.integers(opening_plies + 1))
    game = CaroGame()
    actions = []
    samples = []

    while not game.is_terminal():
        if game.move_count < opening_count:
            choices = candidate_actions(game.board, agent.RADIUS)
            action = choices[int(rng.integers(len(choices)))]
        else:
            action = agent.choose_action(game.board, game.current_player)
            state = normalize_board(game.board, game.current_player)
            row, col = action
            samples.append(Sample(state, row * BOARD_SIZE + col, game_id, game.move_count + 1))
        game.step(action)
        actions.append(action)

    return {"game_id": game_id, "seed": seed, "split": split, "opening_plies": opening_count,
            "moves": actions, "result": game.result()}, samples


def _write_split(path: Path, samples: list[Sample], seen: set[bytes]) -> dict[str, int]:
    retained = []
    for sample in samples:
        key = canonical_key(sample.board)
        if key not in seen:
            seen.add(key)
            retained.append(sample)
    if not retained:
        raise ValueError(f"Tập {path.stem} rỗng sau khi khử trùng lặp")

    np.savez_compressed(
        path,
        boards=np.stack([sample.board for sample in retained]),
        actions=np.array([sample.action for sample in retained], dtype=np.int16),
        game_ids=np.array([sample.game_id for sample in retained], dtype=np.int32),
        plies=np.array([sample.ply for sample in retained], dtype=np.uint8),
    )
    return {"samples_before_dedup": len(samples), "samples": len(retained),
            "removed_duplicates": len(samples) - len(retained),
            "games_with_samples": len({sample.game_id for sample in retained})}


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def generate_dataset(output: Path, *, games: int, seed: int, opening_plies: int = 4) -> dict:
    """Chơi từ đầu, tách theo ván, khử trùng và ghi bộ dữ liệu vào thư mục mới."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Thư mục đích đã tồn tại: {output}")
    if games < 3 or seed < 0 or not 0 <= opening_plies <= 2 * WIN_LENGTH - 2:
        raise ValueError("Cần ít nhất 3 ván, seed không âm và số nước mở đầu trong khoảng 0–8")
    output.parent.mkdir(parents=True, exist_ok=True)
    split_seed, *game_seeds = np.random.SeedSequence(seed).spawn(games + 1)
    assignments = _assign_splits(games, np.random.default_rng(split_seed))
    samples: dict[str, list[Sample]] = {split: [] for split in SPLITS}
    outcomes = {"X": 0, "O": 0, "draw": 0}

    with TemporaryDirectory(dir=output.parent, prefix=".caro-") as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        with (staging / "games.jsonl").open("w", encoding="utf-8", newline="\n") as log:
            for game_id, sequence in enumerate(game_seeds):
                game_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
                split = assignments[game_id]
                record, positions = _play_game(game_id, game_seed, split, opening_plies)
                log.write(json.dumps(record) + "\n")
                samples[split].extend(positions)
                outcomes[{1: "X", -1: "O", 0: "draw"}[record["result"]]] += 1

        # Giữ thế trùng trong test trước, sau đó mới đến val và train.
        seen: set[bytes] = set()
        statistics = {}
        for split in reversed(SPLITS):
            statistics[split] = _write_split(staging / f"{split}.npz", samples[split], seen)
            statistics[split]["games_assigned"] = sum(value == split for value in assignments.values())

        manifest = {
            "schema_version": 1, "teacher": TacticalAgent.VERSION, "radius": TacticalAgent.RADIUS,
            "rules": {"board_size": BOARD_SIZE, "win_length": WIN_LENGTH, "freestyle": True},
            "games": games, "seed": seed, "max_opening_plies": opening_plies, "outcomes": outcomes,
            "encoding": "current_player_relative_int8", "action_encoding": "row_major_zero_based",
            "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1}, "splits": statistics,
            "deduplication": "canonical_D4_keep_test_then_val_then_train",
            "augmentation": "optional_D4_train_only",
            "versions": {"python": platform.python_version(), "numpy": np.__version__},
            "sha256": {path.name: _sha256(path) for path in sorted(staging.iterdir())},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Tự đấu để sinh dữ liệu huấn luyện CNN Policy Caro.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Thư mục đích chưa tồn tại")
    parser.add_argument("--games", type=int, default=1000, help="Số ván tự đấu")
    parser.add_argument("--seed", type=int, default=42, help="Seed tái lập thí nghiệm")
    parser.add_argument("--opening-plies", type=int, default=4, help="Số nước mở đầu ngẫu nhiên tối đa")
    args = parser.parse_args()
    try:
        manifest = generate_dataset(args.output, games=args.games, seed=args.seed, opening_plies=args.opening_plies)
    except (FileExistsError, ValueError) as error:
        parser.exit(2, f"Không thể sinh dữ liệu: {error}\n")
    print(f"Đã sinh {manifest['games']} ván vào {args.output}")
    for split in SPLITS:
        stats = manifest["splits"][split]
        print(f"{split}: {stats['games_assigned']} ván, {stats['samples']} mẫu, "
              f"đã loại {stats['removed_duplicates']} thế trùng")


if __name__ == "__main__":
    main()
