# `.claude/` — persisted assistant context

Version-controlled copy of the Claude Code working context for this project, so it survives
across machines/sessions and is visible to collaborators.

- **`memory/`** — the assistant's long-term memory files (facts, findings, working guidelines).
  `MEMORY.md` is the index; each other file is one atomic fact/finding. Mirror of the local
  `~/.claude/.../memory/` store as of the last commit.
- **`SESSION_LOG.md`** — chronological recap of what was done, decided, and learned in the
  2026-07-25 session (strategy pivot to the pilkwang pipeline + tuning).

The live strategy doc is `docs/STRATEGY.md` (kept current); this folder is the assistant-context
snapshot that complements it.
