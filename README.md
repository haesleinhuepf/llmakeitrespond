# llmakeitrespond

Prompt LLMs on several OpenAI-compatible servers; server selection is transparent.

```
pip install .
export GROQ_API_KEY=... OPENAI_API_KEY=...   # only servers with keys are contacted
```

```python
from llmakeitrespond import prompt
r = prompt("Hi!", "llama-3.3-70b-versatile")
print(r.text, r.model, r.server, r.response_time, r.timings)
```

On first use, model lists are pulled from all servers with a key set (`SERVERS` in
`llmakeitrespond`). `prompt` sends the message to all servers offering the model in
parallel, remembers the fastest, and uses only it afterwards (until restart). If it
fails, all servers are queried again.
