"""Kiểm thử tương tác Tkinter bằng các callback thật của ứng dụng."""

from types import SimpleNamespace
import time
import tkinter as tk

from app.demo import CaroDemo, HUMAN


def _close(app):
    app.close()
    deadline = time.monotonic() + 35
    while not app.closed and time.monotonic() < deadline:
        app.root.update()
        time.sleep(0.01)
    assert app.closed


def test_human_game_rejects_occupied_cell_and_announces_win():
    root = tk.Tk()
    root.withdraw()
    app = CaroDemo(root)
    app.players[-1].set(HUMAN)
    app.start_game()
    deadline = time.monotonic() + 5
    while app.pending is not None and time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)
    assert app.pending is None and app.running
    moves = [(0, 0), (10, 0), (0, 1), (10, 2), (0, 2), (10, 4), (0, 3), (10, 6), (0, 4)]
    try:
        for index, (row, col) in enumerate(moves):
            event = SimpleNamespace(x=app.MARGIN+col*app.CELL, y=app.MARGIN+row*app.CELL)
            app.on_click(event)
            assert app.game.move_count == index + 1
            app.on_click(event)
            assert app.game.move_count == index + 1
        assert app.finished and app.game.result() == 1
        assert "X thắng" in app.status.get()
        app.start_game()
        assert app.game.move_count == 0
    finally:
        _close(app)


def test_neural_demo_move_then_stop():
    root = tk.Tk()
    root.withdraw()
    app = CaroDemo(root)
    app.players[1].set("CNN Policy")
    app.players[-1].set(HUMAN)
    try:
        app.start_game()
        deadline = time.monotonic() + 20
        while app.game.move_count == 0 and app.running and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        assert app.game.move_count == 1 and not app.finished
        assert app.game.current_player == -1
        app.stop_game()
        app.on_click(SimpleNamespace(x=app.MARGIN, y=app.MARGIN))
        assert app.game.move_count == 1
    finally:
        _close(app)


def test_close_during_agent_startup():
    root = tk.Tk()
    root.withdraw()
    app = CaroDemo(root)
    app.players[1].set("CNN Policy")
    app.start_game()
    _close(app)
    assert app.pending is None and not app.workers
