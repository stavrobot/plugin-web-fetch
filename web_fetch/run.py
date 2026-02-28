#!/usr/bin/env -S uv run
# /// script
# dependencies = []
# ///

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


KNOWN_PARAMETERS = {"url", "task"}

MAX_CONTENT_CHARACTERS = 400_000

SYSTEM_PROMPT = (
    "You are a web page analysis assistant. You will receive the text content "
    "of a web page and a task describing what to do with it. Execute the task "
    "against the page content and return your result. Be precise and thorough."
)


def strip_html(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_url(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Stavrobot/1.0)"},
    )
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8", errors="replace")


def call_anthropic(api_key: str, model: str, url: str, task: str, page_text: str) -> str:
    user_message = f"Task: {task}\n\nPage URL: {url}\n\nPage content:\n{page_text}"
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 4096,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }
    ).encode()

    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        print(f"Anthropic API error {error.code}: {error_body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as error:
        print(f"Network error calling Anthropic API: {error.reason}", file=sys.stderr)
        sys.exit(1)

    text_blocks = [block["text"] for block in result["content"] if block["type"] == "text"]
    return "".join(text_blocks)


def main() -> None:
    parameters = json.load(sys.stdin)
    unknown = set(parameters) - KNOWN_PARAMETERS
    if unknown:
        print(f"Unknown parameters: {', '.join(sorted(unknown))}", file=sys.stderr)
        sys.exit(1)

    if "url" not in parameters:
        print("Missing required parameter: url", file=sys.stderr)
        sys.exit(1)
    if "task" not in parameters:
        print("Missing required parameter: task", file=sys.stderr)
        sys.exit(1)

    if not isinstance(parameters["url"], str):
        print("Parameter 'url' must be a string", file=sys.stderr)
        sys.exit(1)
    if not isinstance(parameters["task"], str):
        print("Parameter 'task' must be a string", file=sys.stderr)
        sys.exit(1)

    config = json.loads(Path("../config.json").read_text())
    api_key: str = config["api_key"]
    model: str = config["model"]

    url: str = parameters["url"]
    task: str = parameters["task"]

    print(f"[stavrobot] web_fetch fetching URL: {url}", file=sys.stderr)

    try:
        html = fetch_url(url)
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code} fetching {url}: {error.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as error:
        print(f"Network error fetching {url}: {error.reason}", file=sys.stderr)
        sys.exit(1)

    page_text = strip_html(html)

    if len(page_text) > MAX_CONTENT_CHARACTERS:
        print(
            f"[stavrobot] web_fetch truncating page content from {len(page_text)} to {MAX_CONTENT_CHARACTERS} characters",
            file=sys.stderr,
        )
        page_text = page_text[:MAX_CONTENT_CHARACTERS]

    print(f"[stavrobot] web_fetch fetched {len(page_text)} characters of text from {url}", file=sys.stderr)
    print(f"[stavrobot] web_fetch calling LLM (model: {model})", file=sys.stderr)

    result = call_anthropic(api_key, model, url, task, page_text)

    print(f"[stavrobot] web_fetch LLM done", file=sys.stderr)

    json.dump({"result": result}, sys.stdout)


main()
