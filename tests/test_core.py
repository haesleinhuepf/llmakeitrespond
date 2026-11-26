import time
import llmakeitrespond as l
from llmakeitrespond import _core


def setup_function(_):
    l.reset()
    _core._models = {"a": ["m"], "b": ["org/m"], "c": ["x"]}


def test_fastest_remembered(monkeypatch):
    calls = []

    def ask(server, mid, msg):
        calls.append(server)
        time.sleep(0.01 if server == "a" else 0.1)
        return "hi", 0.01 if server == "a" else 0.1

    monkeypatch.setattr(_core, "_ask", ask)
    r = l.prompt("hello", "m")
    assert r.server == "a" and r.text == "hi" and set(r.timings) == {"a", "b"}
    calls.clear()
    assert l.prompt("hello", "m").server == "a"
    assert calls == ["a"]


def test_failure_falls_back(monkeypatch):
    def ask(server, mid, msg):
        if server == "a":
            raise RuntimeError
        return "ok", 0.1

    monkeypatch.setattr(_core, "_ask", ask)
    assert l.prompt("x", "m").server == "b"
    assert l.prompt("x", "m").server == "b"
