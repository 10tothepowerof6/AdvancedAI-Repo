"""Runner từng ván với tiến trình riêng để cưỡng chế timeout trên Windows."""

import multiprocessing as mp
from multiprocessing.connection import Connection
from time import perf_counter
from typing import Callable

from agents.base_agent import BaseAgent
from config.search_budget import SearchBudget
from core import CaroGame
from evaluation.metrics import MatchRecord

AgentFactory = Callable[[], BaseAgent]


class AgentExecutionError(RuntimeError):
    """Lỗi được truyền rõ từ tiến trình agent về runner."""


def _worker(connection: Connection, factory: AgentFactory, budget: SearchBudget) -> None:
    try:
        # Biên tiến trình phải chuyển lỗi thành thông điệp, không giả lập nước đi thành công.
        try:
            agent = factory()
            agent.budget = budget
        except Exception as error:
            connection.send(("startup_error", f"{type(error).__name__}: {error}"))
            return
        connection.send(("ready",))
        while True:
            request = connection.recv()
            if request is None:
                return
            board, player = request
            try:
                action = agent.choose_action(board, player)
            except Exception as error:
                connection.send(("agent_error", f"{type(error).__name__}: {error}"))
                continue
            connection.send(("move", action, agent.stats.copy()))
    except (EOFError, BrokenPipeError):
        # Tiến trình cha đã đóng kết nối hoặc kết thúc ván.
        return
    finally:
        connection.close()


class AgentWorker:
    def __init__(self, factory: AgentFactory, budget: SearchBudget,
                 startup_timeout_s: float = 30.0):
        self.budget = budget
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, factory, budget))
        self.process.start()
        child.close()
        try:
            if not self.connection.poll(startup_timeout_s):
                raise AgentExecutionError("Khởi tạo agent quá hạn; chưa tính là ván đấu")
            response = self.connection.recv()
            if response[0] != "ready":
                raise AgentExecutionError(f"Không khởi tạo được agent: {response[1]}")
        except BaseException:
            self.close()
            raise

    def choose_action(self, board, player: int, limit_s: float):
        """Đo trọn yêu cầu/đáp ứng; khởi tạo agent nằm ngoài ngân sách mỗi nước."""
        start = perf_counter()
        try:
            self.connection.send((board, player))
            remaining = max(0.0, limit_s - (perf_counter() - start))
            if not self.connection.poll(remaining):
                raise TimeoutError("Agent vượt giới hạn thời gian mỗi nước")
            response = self.connection.recv()
        except TimeoutError:
            raise
        except (EOFError, BrokenPipeError, OSError) as error:
            raise AgentExecutionError("Tiến trình agent ngừng phản hồi") from error
        elapsed = perf_counter() - start
        if elapsed > limit_s:
            raise TimeoutError("Kết quả được nhận sau giới hạn thời gian mỗi nước")
        if response[0] == "agent_error":
            raise AgentExecutionError(response[1])
        return response[1], response[2], elapsed

    def close(self) -> None:
        # Kết thúc tiến trình riêng, không để agent quá hạn tiếp tục chạy nền.
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.connection.close()


def play_match(factory_x: AgentFactory, factory_o: AgentFactory, *,
               agent_x: str, agent_o: str, budget: SearchBudget | None = None,
               game: CaroGame | None = None, safety_timeout_s: float = 30.0
               ) -> tuple[MatchRecord, list[dict[str, object]]]:
    """Trả kết quả và log từng yêu cầu; lỗi cấu hình/khởi tạo được phát sinh ra ngoài.

    Factory phải import được ở module .py; script gọi runner cần bảo vệ điểm vào
    bằng __main__ trên Windows. Mỗi agent tồn tại xuyên suốt một ván để giữ RNG.
    """
    budget = budget if budget is not None else SearchBudget()
    # Dùng chính validation của ngân sách để kiểm tra watchdog.
    SearchBudget(time_limit_s=safety_timeout_s)
    MatchRecord(agent_x, agent_o, 0)
    state = game.clone() if game is not None else CaroGame()
    if state.is_terminal():
        raise ValueError("Không khởi chạy ván đấu từ trạng thái đã kết thúc")
    limit = budget.time_limit_s if budget.mode == "time" else safety_timeout_s
    workers: dict[int, AgentWorker] = {}
    logs: list[dict[str, object]] = []
    try:
        workers[1] = AgentWorker(factory_x, budget)
        workers[-1] = AgentWorker(factory_o, budget)
        while not state.is_terminal():
            player = state.current_player
            start = perf_counter()
            log: dict[str, object] = {"player": player, "ply": state.move_count + 1,
                                      "mode": budget.mode, "limit_s": limit,
                                      "budget": {"simulations": budget.simulations,
                                                 "depth": budget.depth,
                                                 "time_limit_s": budget.time_limit_s}}
            try:
                action, stats, elapsed = workers[player].choose_action(state.board, player, limit)
            except (TimeoutError, AgentExecutionError) as error:
                reason = "timeout" if isinstance(error, TimeoutError) else "agent_error"
                log.update(status=reason, error=str(error), elapsed_s=perf_counter() - start)
                logs.append(log)
                return MatchRecord(agent_x, agent_o, -player, reason), logs
            try:
                state.step(action)
            except ValueError as error:
                log.update(status="illegal_action", error=str(error), elapsed_s=elapsed)
                logs.append(log)
                return MatchRecord(agent_x, agent_o, -player, "illegal_action"), logs
            log.update(status="move", action=action, stats=stats, elapsed_s=elapsed)
            logs.append(log)
        result = state.result()
        if result is None:
            raise RuntimeError("Trạng thái kết thúc thiếu kết quả")
        return MatchRecord(agent_x, agent_o, result), logs
    finally:
        for worker in workers.values():
            worker.close()
