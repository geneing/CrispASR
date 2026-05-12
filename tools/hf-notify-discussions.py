#!/usr/bin/env python3
"""Poll HuggingFace Hub for new / recently-updated discussions across
every repo owned by a given user/org.

Why this exists: HF doesn't email maintainers when someone opens a
discussion or community-post on a repo — but those threads are how
external users actually report bugs (the suneetk glm-asr-nano issue
sat for 23 days before we noticed). A nightly poller closes that
gap.

Usage:

    # one-off, prints to stdout
    python tools/hf-notify-discussions.py --user cstr

    # with a state file so we only see new/updated threads since last run
    python tools/hf-notify-discussions.py --user cstr \\
        --state ~/.cache/crispasr/hf-notify-seen.json

    # only the last 7 days (no state file)
    python tools/hf-notify-discussions.py --user cstr --since-days 7

    # emit JSON for programmatic chaining (slack/email/etc)
    python tools/hf-notify-discussions.py --user cstr --json

Auth: anonymous works for public repos (we own no private ones).
Set HF_TOKEN if you ever change that.

Exit code: number of new threads (0 if nothing new). Lets cron-style
runners gate on "is there anything to look at".

Schedule via macOS launchd (sample plist at the bottom of this
docstring), system cron, GitHub Actions, or as a fourth Kaggle
notebook (`kernel_type: script`, weekly schedule, posts results to
the same channel as the regression suite).

Sample launchd plist (drop in ~/Library/LaunchAgents/):

    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
                            "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
      <key>Label</key><string>com.cstr.hf-notify-discussions</string>
      <key>ProgramArguments</key>
      <array>
        <string>/Users/christianstrobele/miniconda3/bin/python</string>
        <string>/Users/christianstrobele/code/CrispASR/tools/hf-notify-discussions.py</string>
        <string>--user</string><string>cstr</string>
        <string>--state</string>
        <string>/Users/christianstrobele/.cache/crispasr/hf-notify-seen.json</string>
        <string>--out</string>
        <string>/Users/christianstrobele/.cache/crispasr/hf-notify-latest.md</string>
      </array>
      <key>StartCalendarInterval</key>
      <dict>
        <key>Hour</key><integer>9</integer>
        <key>Minute</key><integer>0</integer>
      </dict>
      <key>StandardOutPath</key>
      <string>/Users/christianstrobele/.cache/crispasr/hf-notify.log</string>
      <key>StandardErrorPath</key>
      <string>/Users/christianstrobele/.cache/crispasr/hf-notify.err</string>
    </dict>
    </plist>

Then: `launchctl load ~/Library/LaunchAgents/com.cstr.hf-notify-discussions.plist`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass
class Thread:
    repo_id: str
    repo_type: str          # model | dataset | space
    num: int                # discussion number on HF
    title: str
    author: str
    created_at: datetime
    last_at: datetime       # last activity (new comment or status change)
    is_pull_request: bool
    status: str             # "open" / "closed"

    @property
    def url(self) -> str:
        prefix = "" if self.repo_type == "model" else f"{self.repo_type}s/"
        kind = "discussions" if not self.is_pull_request else "discussions"
        return f"https://huggingface.co/{prefix}{self.repo_id}/{kind}/{self.num}"

    @property
    def key(self) -> str:
        return f"{self.repo_type}:{self.repo_id}:{self.num}"


def collect_repos(api, user: str) -> list[tuple[str, str]]:
    """Return [(repo_id, repo_type), …] for everything owned by `user`."""
    repos: list[tuple[str, str]] = []
    # `list_models`, `list_datasets`, `list_spaces` all accept `author=`.
    for fn, kind in [
        (api.list_models, "model"),
        (api.list_datasets, "dataset"),
        (api.list_spaces, "space"),
    ]:
        try:
            for r in fn(author=user, full=False):
                # The id attribute is `cstr/foo` for owned repos.
                repo_id = getattr(r, "id", None) or getattr(r, "modelId", None)
                if repo_id:
                    repos.append((repo_id, kind))
        except Exception as exc:
            print(f"# warn: failed to list {kind}s for {user}: {exc}",
                  file=sys.stderr)
    return repos


def collect_threads(api, repos: list[tuple[str, str]]) -> list[Thread]:
    """Fetch all discussion threads across `repos`. Anonymous OK for public."""
    threads: list[Thread] = []
    for repo_id, repo_type in repos:
        try:
            discs = api.get_repo_discussions(
                repo_id=repo_id, repo_type=repo_type)
            for d in discs:
                # `d` is a DiscussionWithDetails; HF's API exposes:
                #   num, title, status, author, created_at, is_pull_request
                last = d.created_at
                # Some HF clients expose `events` with timestamps; if the
                # discussion has any new events, take the max. Otherwise
                # fall back to created_at.
                events = getattr(d, "events", None)
                if events:
                    try:
                        last = max(getattr(ev, "created_at", d.created_at)
                                   for ev in events)
                    except Exception:
                        pass
                threads.append(Thread(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    num=d.num,
                    title=d.title,
                    author=getattr(d, "author", "?"),
                    created_at=d.created_at,
                    last_at=last,
                    is_pull_request=getattr(d, "is_pull_request", False),
                    status=getattr(d, "status", "open"),
                ))
        except Exception as exc:
            # Some repos throw if discussions are disabled. Tolerate.
            print(f"# warn: discussions for {repo_id} ({repo_type}): {exc}",
                  file=sys.stderr)
    return threads


def load_state(state_path: Path | None) -> dict[str, str]:
    """{thread.key: iso last_at} of what we've already reported."""
    if not state_path or not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {}


def save_state(state_path: Path | None, threads: list[Thread]) -> None:
    if not state_path:
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {t.key: t.last_at.isoformat() for t in threads}
    state_path.write_text(json.dumps(payload, indent=1))


def is_new(t: Thread, state: dict[str, str], since: datetime | None) -> bool:
    """A thread is 'new' if either:
      - it's missing from state, OR
      - state has an older last_at than what HF reports now.
    If `since` is given, also drop threads older than that as a coarse
    filter (useful for first-run without state).
    """
    if since and t.last_at < since:
        return False
    prev = state.get(t.key)
    if prev is None:
        return True
    try:
        prev_dt = datetime.fromisoformat(prev)
    except Exception:
        return True
    return t.last_at > prev_dt


def render_markdown(new_threads: list[Thread]) -> str:
    if not new_threads:
        return "_no new HuggingFace discussions since last check._\n"
    out = [f"# HuggingFace discussions — {len(new_threads)} new / updated\n"]
    new_threads.sort(key=lambda t: t.last_at, reverse=True)
    for t in new_threads:
        emoji = "🟢" if t.status == "open" else "✅"
        kind = "PR" if t.is_pull_request else "discussion"
        out.append(
            f"## {emoji} [{t.repo_id} #{t.num} — {t.title}]({t.url})\n"
            f"  - {kind}, status: **{t.status}**, by `{t.author}`\n"
            f"  - opened {t.created_at:%Y-%m-%d %H:%M} UTC, "
            f"last activity {t.last_at:%Y-%m-%d %H:%M} UTC\n"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Poll HF Hub for new discussions across cstr/*")
    parser.add_argument("--user", default="cstr",
                        help="HF user/org whose repos to poll (default: cstr)")
    parser.add_argument("--state",
                        help="JSON file storing what we've already reported")
    parser.add_argument("--since-days", type=int,
                        help="Only report threads updated in the last N days "
                             "(applied even when --state is given, as a "
                             "safety bound)")
    parser.add_argument("--out",
                        help="Write the markdown summary here in addition "
                             "to stdout")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of markdown")
    parser.add_argument("--include-prs", action="store_true",
                        help="Also include pull requests (default: "
                             "discussions only)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. "
              "Run: pip install huggingface_hub", file=sys.stderr)
        return 2

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    print(f"# polling HF for discussions on {args.user}/*", file=sys.stderr)
    repos = collect_repos(api, args.user)
    print(f"# {len(repos)} repos found", file=sys.stderr)

    threads = collect_threads(api, repos)
    if not args.include_prs:
        threads = [t for t in threads if not t.is_pull_request]
    print(f"# {len(threads)} total discussions across them", file=sys.stderr)

    state_path = Path(args.state).expanduser() if args.state else None
    state = load_state(state_path)

    since = None
    if args.since_days:
        since = datetime.now(timezone.utc) - timedelta(days=args.since_days)

    new_threads = [t for t in threads if is_new(t, state, since)]

    if args.json:
        payload = [{
            "repo_id": t.repo_id,
            "repo_type": t.repo_type,
            "num": t.num,
            "title": t.title,
            "author": t.author,
            "status": t.status,
            "created_at": t.created_at.isoformat(),
            "last_at": t.last_at.isoformat(),
            "url": t.url,
        } for t in new_threads]
        rendered = json.dumps(payload, indent=2)
    else:
        rendered = render_markdown(new_threads)

    print(rendered)
    if args.out:
        Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).expanduser().write_text(rendered)

    save_state(state_path, threads)

    return len(new_threads)


if __name__ == "__main__":
    sys.exit(main())
