"""Server/model selection for OpenAI-compatible LLM servers."""
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# server name -> (base_url, environment variable holding the API key)
SERVERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
}

_lock = threading.Lock()
_models: Optional[Dict[str, List[str]]] = None  # server -> model ids
_fastest: Dict[str, str] = {}  # model query -> server


@dataclass
class Response:
    """Answer of a model, including how it was obtained."""
    text: str
    model: str
    server: str
    response_time: float
    timings: Dict[str, float] = field(default_factory=dict)  # server -> seconds (parallel probes)

    def __str__(self):
        return self.text


def _client(server):
    """Return an OpenAI client for a server, or None if its key is missing."""
    base_url, env = SERVERS[server]
    key = os.environ.get(env)
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=key)


def _active_servers():
    return [s for s in SERVERS if os.environ.get(SERVERS[s][1])]


def refresh() -> Dict[str, List[str]]:
    """(Re-)pull model lists from all servers whose API key is set."""
    global _models
    result = {}

    def fetch(server):
        return server, sorted(m.id for m in _client(server).models.list())

    servers = _active_servers()
    if servers:
        with ThreadPoolExecutor(len(servers)) as ex:
            futures = [ex.submit(fetch, s) for s in servers]
            for f in futures:
                try:
                    server, ids = f.result()
                    result[server] = ids
                except Exception:
                    pass  # unreachable server: ignore
    with _lock:
        _models = result
    return result


def available_models() -> Dict[str, List[str]]:
    """Models per server (fetched lazily on first use)."""
    if _models is None:
        refresh()
    return dict(_models)


def reset():
    """Forget model lists and remembered fastest servers."""
    global _models
    with _lock:
        _models = None
        _fastest.clear()


def _matching(model):
    """Return {server: exact model id} for models matching `model`.

    Exact (case-insensitive) matches win; otherwise the id's last path
    component or a substring is accepted.
    """
    q = model.lower()
    found = {}
    for server, ids in available_models().items():
        exact = [i for i in ids if i.lower() == q or i.lower().split("/")[-1] == q]
        if not exact:
            exact = [i for i in ids if q in i.lower()]
        if exact:
            found[server] = min(exact, key=len)
    return found


def _ask(server, model_id, message):
    start = time.perf_counter()
    r = _client(server).chat.completions.create(
        model=model_id, messages=[{"role": "user", "content": message}])
    return r.choices[0].message.content, time.perf_counter() - start


def prompt(message: str, model: str) -> Response:
    """Prompt `model`. The fastest server is chosen transparently.

    The first call queries all servers offering the model in parallel and
    remembers the fastest one; later calls only use it, until it fails.
    """
    candidates = _matching(model)
    if not candidates:
        raise ValueError(f"No available server offers a model matching '{model}'. "
                         f"Set API keys, e.g. {', '.join(v[1] for v in SERVERS.values())}.")

    server = _fastest.get(model)
    if server in candidates:
        try:
            text, t = _ask(server, candidates[server], message)
            return Response(text, candidates[server], server, t, {server: t})
        except Exception:
            _fastest.pop(model, None)

    results, errors = {}, {}
    with ThreadPoolExecutor(len(candidates)) as ex:
        futures = {ex.submit(_ask, s, mid, message): s for s, mid in candidates.items()}
        for f in as_completed(futures):
            s = futures[f]
            try:
                results[s] = f.result()
            except Exception as e:
                errors[s] = e
    if not results:
        raise RuntimeError(f"All servers failed for '{model}': {errors}")
    best = min(results, key=lambda s: results[s][1])
    _fastest[model] = best
    text, t = results[best]
    return Response(text, candidates[best], best, t, {s: r[1] for s, r in results.items()})
