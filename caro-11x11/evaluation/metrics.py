"""W/D/L luôn tính từ góc nhìn agent được yêu cầu, không mặc định theo X."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Iterable, Literal

import numpy as np


@dataclass(frozen=True)
class MatchRecord:
    agent_x: str
    agent_o: str
    winner: int
    reason: Literal["completed", "timeout", "agent_error", "illegal_action"] = "completed"

    def __post_init__(self) -> None:
        if (not isinstance(self.agent_x, str) or not isinstance(self.agent_o, str)
                or not self.agent_x or not self.agent_o or self.agent_x == self.agent_o):
            raise ValueError("Hai cấu hình thi đấu phải có định danh khác nhau và không rỗng")
        if (isinstance(self.winner, bool) or not isinstance(self.winner, Integral)
                or self.winner not in (-1, 0, 1)):
            raise ValueError("Kết quả phải là -1, 0 hoặc 1")
        if self.reason not in ("completed", "timeout", "agent_error", "illegal_action"):
            raise ValueError("Nguyên nhân kết thúc không hợp lệ")
        if self.reason != "completed" and self.winner == 0:
            raise ValueError("Ván xử thua không được ghi thành hòa")


def summarize_results(records: Iterable[MatchRecord], agent: str) -> dict[str, object]:
    """Tính cả ván xử thua vào W/D/L; thống kê lỗi do chính agent gây ra riêng."""
    wins = draws = losses = 0
    forfeits = {"timeout": 0, "agent_error": 0, "illegal_action": 0}
    for record in records:
        if agent not in (record.agent_x, record.agent_o):
            continue
        perspective = 1 if agent == record.agent_x else -1
        outcome = record.winner * perspective
        wins += int(outcome == 1)
        draws += int(outcome == 0)
        losses += int(outcome == -1)
        if outcome == -1 and record.reason != "completed":
            forfeits[record.reason] += 1
    total = wins + draws + losses
    return {"wins": wins, "draws": draws, "losses": losses, "total": total,
            "score": (wins + 0.5 * draws) / total if total else None,
            "forfeits": forfeits}


def summarize_latencies(seconds: Iterable[float]) -> dict[str, float | int | None]:
    """Đầu vào là thời gian nước hợp lệ hoàn thành, đơn vị giây; p95 nội suy tuyến tính."""
    values = list(seconds)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not isfinite(v) or v < 0 for v in values):
        raise ValueError("Thời gian phải là số hữu hạn không âm")
    return {"count": len(values),
            "mean_s": float(np.mean(values)) if values else None,
            "p95_s": float(np.percentile(values, 95, method="linear")) if values else None}
