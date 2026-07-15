# diag_anthropic.py -- temporary, delete after use
from dotenv import load_dotenv
load_dotenv()
import anthropic, os

client = anthropic.Anthropic()
resp = client.messages.create(
    model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
    max_tokens=20,
    temperature=0.0,
    system='Respond with ONLY this exact JSON: {"ok": true}',
    messages=[{"role": "user", "content": "ping"}],
)
print("stop_reason:", resp.stop_reason)
print("content blocks:", resp.content)