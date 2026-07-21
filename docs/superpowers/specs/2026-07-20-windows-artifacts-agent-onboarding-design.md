# Windows Artifact Writes and Agent Onboarding Design

## Goal

Make the deterministic offline examples preserve exact artifact bytes on Windows, while keeping POSIX behavior unchanged, and provide concise repository instructions for coding agents.

## Artifact Write Behavior

`ArtifactStore._write_bytes()` will keep its existing exclusive creation, write loop, and file synchronization. Its open flags will include `getattr(os, "O_BINARY", 0)`. Windows will then write supplied bytes without newline conversion. POSIX platforms do not define `os.O_BINARY`; the fallback is zero and preserves their current flags and behavior.

The regression test will write an LF-containing byte payload through `_write_bytes()` and assert `Path.read_bytes()` returns exactly the original payload. It will fail on current Windows because the text-mode descriptor stores CRLF, and pass on Windows after the change as well as on POSIX before and after the change.

## Agent Onboarding

Add root `AGENTS.md` as the repository-level instruction entry point. It will link to the existing README, contribution, security, and product documentation rather than duplicating them. It will define the product/reproduction boundary, offline-first and no-live-credential defaults, sensitive-data restrictions, expected validation, and pull-request handoff requirements.

## Scope and Non-Goals

No public API, artifact schema, provider behavior, benchmark task selection, or dependency versions will change. The document will be plain Markdown, not a Codex-specific plugin or skill package.
