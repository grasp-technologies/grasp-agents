"""Test whether Anthropic needs web_fetch content in round-trip."""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from anthropic import AsyncAnthropic


async def main():
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Turn 1: fetch a page
    msg1 = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=[
            {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 2},
        ],
        messages=[
            {"role": "user", "content": "Fetch https://httpbin.org/html and summarize it briefly."}
        ],
    )

    print("=== Turn 1 ===")
    print(f"Stop reason: {msg1.stop_reason}")
    for i, b in enumerate(msg1.content):
        print(f"  Block {i}: type={b.type}")
    print()

    # Turn 2: follow-up using full content (should work)
    print("=== Turn 2a: full content round-trip ===")
    try:
        msg2a = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[
                {"role": "user", "content": "Fetch https://httpbin.org/html and summarize it briefly."},
                {"role": "assistant", "content": msg1.content},
                {"role": "user", "content": "What was the author's name mentioned on that page?"},
            ],
        )
        print(f"  OK - {msg2a.content[0].text[:200]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

    # Turn 2b: strip web_fetch_tool_result content entirely (replace with minimal)
    print("=== Turn 2b: without web_fetch blocks ===")
    stripped = [b for b in msg1.content if b.type not in ("web_fetch_tool_result", "server_tool_use", "code_execution_tool_result")]
    if not stripped:
        stripped = msg1.content  # fallback if everything was stripped
        print("  (nothing left after stripping, using original)")
    try:
        msg2b = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[
                {"role": "user", "content": "Fetch https://httpbin.org/html and summarize it briefly."},
                {"role": "assistant", "content": stripped},
                {"role": "user", "content": "What was the author's name mentioned on that page?"},
            ],
        )
        print(f"  OK - {msg2b.content[0].text[:200]}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")


asyncio.run(main())
