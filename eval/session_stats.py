"""Counts in the Claude Code session log for the AI-usage disclosure: user prompts typed by the student (not tool
results), assistant messages, tool calls by tool name, and files written or edited by the agent.
Run: python3 -I eval/session_stats.py <session.jsonl>"""
import json, sys, collections
user = asst = queued = 0; tools = collections.Counter(); files = set()
for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except ValueError:
        continue
    m = e.get("message") or {}
    c = m.get("content")
    if e.get("type") == "user" and not e.get("isMeta"):
        if isinstance(c, str) or (isinstance(c, list) and any(b.get("type") == "text" for b in c if isinstance(b, dict))):
            user += 1
    if e.get("type") == "attachment" and (e.get("attachment") or {}).get("type") == "queued_command" \
            and "task-notification" not in str(e["attachment"].get("prompt", "")):
        queued += 1  # a message the student typed while the agent was working
    if e.get("type") == "assistant":
        asst += 1
        for b in c or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                tools[b["name"]] += 1
                fp = (b.get("input") or {}).get("file_path")
                if fp and b["name"] in ("Write", "Edit"):
                    files.add(fp.split("exam quiz/")[-1])
print(f"session {sys.argv[1].split('/')[-1]}: user prompts {user} + {queued} typed mid-turn, assistant messages {asst}, tool calls {sum(tools.values())} {dict(tools.most_common())}")
print("files written/edited by the agent:", sorted(files))
