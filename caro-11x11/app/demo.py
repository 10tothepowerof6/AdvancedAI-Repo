"""Giao diện Tkinter chơi người–máy hoặc theo dõi máy–máy."""

from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk

from config.search_budget import SearchBudget
from core.game_engine import CaroGame
from evaluation.match import AgentExecutionError, AgentWorker
from evaluation.registry import AGENT_NAMES, AgentSpec

HUMAN = "Người chơi"
CHOICES = (HUMAN, *AGENT_NAMES.values())
KIND_BY_NAME = {label: kind for kind, label in AGENT_NAMES.items()}


class CaroDemo:
    CELL = 42
    MARGIN = 28

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Caro 11×11 — Trí tuệ nhân tạo nâng cao")
        self.game = CaroGame()
        self.workers = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.closing = False
        self.closed = False
        self.finished = False
        self.running = False
        self.elapsed = 0.0
        self.players = {1: tk.StringVar(value=HUMAN), -1: tk.StringVar(value="CNN Policy")}
        self.mode = tk.StringVar(value="time")
        self.status = tk.StringVar(value="Chọn hai bên rồi bấm Bắt đầu ván mới.")
        panel = ttk.Frame(root, padding=12)
        panel.pack(fill="x")
        for column, (player, symbol) in enumerate(((1, "X — đi trước"), (-1, "O — đi sau"))):
            group = ttk.Frame(panel)
            group.grid(row=0, column=column, padx=5)
            ttk.Label(group, text=symbol).pack(anchor="w")
            ttk.Combobox(group, textvariable=self.players[player], values=CHOICES,
                         state="readonly", width=25).pack()
        ttk.Combobox(panel, textvariable=self.mode, values=("time", "fixed"),
                     state="readonly", width=8).grid(row=1, column=0, pady=8, sticky="w")
        self.start_button = ttk.Button(panel, text="Bắt đầu ván mới", command=self.start_game)
        self.start_button.grid(row=1, column=1, pady=8)
        self.stop_button = ttk.Button(panel, text="Dừng ván", command=self.stop_game)
        self.stop_button.grid(row=2, column=1)
        ttk.Label(panel, text="time: 2 giây/nước | fixed: 1.000 mô phỏng, độ sâu 2").grid(
            row=3, column=0, columnspan=2, pady=5)
        size = self.MARGIN * 2 + self.CELL * 10
        self.canvas = tk.Canvas(root, width=size, height=size, background="#eed6a8", highlightthickness=0)
        self.canvas.pack(padx=12, pady=8)
        self.canvas.bind("<Button-1>", self.on_click)
        ttk.Label(root, textvariable=self.status, wraplength=size, padding=12).pack(fill="x")
        self.draw()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def draw(self):
        self.canvas.delete("all")
        start, end = self.MARGIN, self.MARGIN + 10 * self.CELL
        for index in range(11):
            position = start + index * self.CELL
            self.canvas.create_line(start, position, end, position, fill="#886a3e")
            self.canvas.create_line(position, start, position, end, fill="#886a3e")
        for row, cells in enumerate(self.game.board):
            for col, player in enumerate(cells):
                if player:
                    self.canvas.create_text(start + col * self.CELL, start + row * self.CELL,
                                            text="X" if player == 1 else "O", font=("Consolas", 23, "bold"),
                                            fill="#9b2727" if player == 1 else "#154c84")
        if self.game.last_action is not None:
            row, col = self.game.last_action
            x, y = start + col * self.CELL, start + row * self.CELL
            self.canvas.create_rectangle(x-18, y-18, x+18, y+18, outline="#24834b", width=2)

    def _prepare_workers(self):
        for worker in self.workers.values():
            worker.close()
        self.workers = {}
        try:
            for player, label in self.selected.items():
                if label != HUMAN:
                    self.workers[player] = AgentWorker(AgentSpec(KIND_BY_NAME[label], 42), self.budget)
        except (AgentExecutionError, OSError):
            for worker in self.workers.values():
                worker.close()
            self.workers = {}
            raise

    def start_game(self):
        if self.pending is not None:
            return
        self.selected = {player: value.get() for player, value in self.players.items()}
        self.budget = SearchBudget(mode=self.mode.get())
        self.game.reset()
        self.finished = False
        self.running = True
        self.draw()
        self.start_button.configure(state="disabled")
        self.status.set("Đang khởi tạo tác tử…")
        self.pending = self.executor.submit(self._prepare_workers)
        self.root.after(40, self._poll_start)

    def _poll_start(self):
        if self.closing:
            return
        if not self.pending.done():
            self.root.after(40, self._poll_start)
            return
        future, self.pending = self.pending, None
        self.start_button.configure(state="normal")
        try:
            future.result()
        except (AgentExecutionError, OSError) as error:
            self.running = False
            self.status.set(f"Không khởi tạo được ván: {error}")
            return
        self.next_turn()

    def next_turn(self):
        if self.finished or not self.running or self.pending is not None:
            return
        symbol = "X" if self.game.current_player == 1 else "O"
        if self.game.is_terminal():
            self.finished = True
            result = self.game.result()
            self.status.set("Ván hòa." if result == 0 else f"{'X' if result == 1 else 'O'} thắng!")
            self.start_button.configure(state="normal")
            return
        player = self.game.current_player
        self.status.set(f"Lượt {symbol} — {self.selected[player]} | nước trước {self.elapsed:.3f} giây")
        if player in self.workers:
            self.start_button.configure(state="disabled")
            limit = self.budget.time_limit_s if self.budget.mode == "time" else 30.0
            self.pending = self.executor.submit(self.workers[player].choose_action,
                                                 self.game.board.copy(), player, limit)
            self.root.after(40, self._poll_move)

    def _poll_move(self):
        if self.closing:
            return
        if not self.pending.done():
            self.root.after(40, self._poll_move)
            return
        future, self.pending = self.pending, None
        self.start_button.configure(state="normal")
        try:
            action, _, self.elapsed = future.result()
            if not self.running:
                return
            self.game.step(action)
        except (TimeoutError, AgentExecutionError, ValueError) as error:
            self.finished = True
            symbol = "X" if self.game.current_player == 1 else "O"
            self.status.set(f"{symbol} bị xử thua: {error}")
            return
        self.draw()
        self.root.after(180, self.next_turn)

    def on_click(self, event):
        if not self.running or self.finished or self.pending is not None or self.game.current_player in self.workers:
            return
        col = round((event.x - self.MARGIN) / self.CELL)
        row = round((event.y - self.MARGIN) / self.CELL)
        if not (0 <= row < 11 and 0 <= col < 11) or self.game.board[row, col] != 0:
            self.status.set("Hãy chọn một giao điểm trống trong bàn cờ.")
            return
        self.elapsed = 0.0
        self.game.step((row, col))
        self.draw()
        self.next_turn()

    def stop_game(self):
        self.running = False
        self.status.set("Đã dừng ván. Có thể bắt đầu ván mới sau khi lượt hiện tại kết thúc.")

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.running = False
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.status.set("Đang đóng tác tử và kết thúc ván…")
        self.cleanup = self.executor.submit(self._close_workers)
        self.root.after(40, self._poll_close)

    def _close_workers(self):
        for worker in self.workers.values():
            worker.close()
        self.workers = {}

    def _poll_close(self):
        if not self.cleanup.done():
            self.root.after(40, self._poll_close)
            return
        try:
            self.cleanup.result()
        finally:
            # Chờ luồng nhả mọi tham chiếu tới ứng dụng trước khi hủy đối tượng Tk.
            self.executor.shutdown(wait=True)
            self.pending = None
            self.closed = True
            self.root.destroy()


def main():
    root = tk.Tk()
    CaroDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
