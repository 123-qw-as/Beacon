from math_agent.llm_worker import _worker_main


class _ClosingConnection:
    def __init__(self):
        self.messages = []
        self._requests = [
            {
                "type": "req",
                "payload": {
                    "model": "test",
                    "messages": [{"role": "user", "content": "x"}],
                },
            }
        ]

    def send(self, payload):
        if payload.get("type") == "ready":
            self.messages.append(payload)
            return
        raise BrokenPipeError("parent closed")

    def recv(self):
        return self._requests.pop(0)


def test_worker_exits_quietly_when_parent_closes_pipe(monkeypatch):
    monkeypatch.setattr(
        "math_agent.llm_worker.litellm.completion",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("provider failed")),
    )
    connection = _ClosingConnection()
    _worker_main(connection)
    assert connection.messages == [{"type": "ready"}]
