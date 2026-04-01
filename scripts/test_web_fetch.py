"""Quick script to see what Anthropic web_fetch actually returns."""

import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv()

from anthropic import AsyncAnthropic


def print_block(i, block):
    print(f"--- Block {i}: type={block.type} ---")

    if block.type == "text":
        print(f"  text: {block.text[:300]}")

    elif block.type == "server_tool_use":
        print(f"  id: {block.id}")
        print(f"  name: {block.name}")
        print(f"  input: {json.dumps(block.input, indent=2)}")

    elif block.type == "web_fetch_tool_result":
        print(f"  tool_use_id: {block.tool_use_id}")
        print(f"  caller: {block.caller}")
        content = block.content
        print(f"  content type: {content.type}")

        if content.type == "web_fetch_result":
            print(f"  url: {content.url}")
            print(f"  retrieved_at: {content.retrieved_at}")
            doc = content.content  # DocumentBlock
            print(f"  doc.type: {doc.type}")
            print(f"  doc.title: {doc.title}")
            src = doc.source
            print(f"  source.type: {src.type}")
            print(f"  source.media_type: {src.media_type}")
            print(f"  source.data length: {len(src.data)}")
            print(f"  source.data[:500]: {src.data[:500]}")

        elif content.type == "web_fetch_tool_result_error":
            print(f"  error_code: {content.error_code}")

    else:
        print(f"  (raw): {block}")

    print()


async def main():
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("=== PDF TEST ===\n")

    msg = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        tools=[
            {
                "type": "web_fetch_20260209",
                "name": "web_fetch",
                "max_uses": 2,
            },
        ],
        messages=[
            {
                "role": "user",
                "content": "Fetch https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf and tell me what it contains.",
            }
        ],
    )

    print("Stop reason:", msg.stop_reason)
    print("Usage:", msg.usage)
    print()
    for i, block in enumerate(msg.content):
        print_block(i, block)


asyncio.run(main())
