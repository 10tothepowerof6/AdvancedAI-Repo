"""Ngân sách chung; các giá trị khởi đầu chưa phải cấu hình tối ưu thực nghiệm."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Literal


@dataclass(frozen=True)
class SearchBudget:
    mode: Literal["fixed", "time"] = "fixed"
    simulations: int = 1000
    depth: int = 2
    time_limit_s: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "time"):
            raise ValueError("Chế độ ngân sách phải là fixed hoặc time")
        for name in ("simulations", "depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} phải là số nguyên dương")
        if (isinstance(self.time_limit_s, bool)
                or not isinstance(self.time_limit_s, (int, float))
                or not isfinite(self.time_limit_s) or self.time_limit_s <= 0):
            raise ValueError("Thời gian mỗi nước phải là số hữu hạn dương")
