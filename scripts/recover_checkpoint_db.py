"""
Recover a corrupted muphrid checkpoint DB.

Usage:
    uv run python scripts/recover_checkpoint_db.py [<db_path>]

Defaults:
    db_path = checkpoints.db

What it does:
    1. Runs `PRAGMA integrity_check`. Exits if the DB is already healthy.
    2. Recovery preference order:
         (a) `sqlite3 .recover` — the canonical SQLite recovery tool. Uses
             the same b-tree page parser that powers SQLite internally and
             produces a truly clean DB. Used when the `sqlite3` CLI is
             available (most macOS/Linux installs ship with it).
         (b) Python row-walk fallback — copies readable rows table-by-table
             via the standard sqlite3 module. Less thorough than `.recover`
             and can produce a DB with lingering b-tree page-orphan issues
             that fail PRAGMA integrity_check on the result, even though
             the row data itself is intact.
       Both paths verify the recovered DB with PRAGMA integrity_check. If
       the result is not "ok" the script aborts WITHOUT touching the
       original file, so a partial recovery never overwrites a known-bad
       file with a different-bad file.
    3. On success: quarantines the broken file as
       <db_path>.corrupt-<UTC-timestamp>, promotes the recovered DB into
       <db_path>, and reports per-table recovery counts.

Common causes of corruption:
  * Python process killed mid-write (SIGKILL, OOM, hard reboot, parent
    Gradio terminated while the agent was streaming an update).
  * Two processes writing concurrently to the same file (e.g. Gradio
    started while the CLI was still running, or two Gradio instances).
  * Disk-level fault (rare but happens).

Why this is a script and not an automatic recovery on every launch:
  Silent self-healing can mask real disk issues. This stays opt-in so the
  operator can inspect the broken file first if they want, then deliberately
  trade it for a recovered one.

Lesson learned (from the run that motivated this script):
  My initial recovery used only the Python row-walk approach. It produced
  a DB whose `PRAGMA integrity_check` reported "ok" right after creation
  but flagged b-tree corruption on subsequent checks — likely because the
  WAL hadn't fully checkpointed and orphan pages survived the close.
  sqlite3 .recover, when available, is strictly better; the row-walk is
  only a fallback for environments without the CLI.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
import shlex
from datetime import datetime, timezone
from pathlib import Path


def _integrity_check(db: Path) -> tuple[bool, list[str]]:
    """Return (ok, issue_list). issue_list is empty when ok."""
    try:
        conn = sqlite3.connect(str(db))
        try:
            rows = conn.execute("PRAGMA integrity_check").fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        return False, [f"connection-level error: {e}"]
    issues = [str(r[0]) for r in rows]
    return issues == ["ok"], (issues if issues != ["ok"] else [])


def _which_sqlite3() -> str | None:
    """Find a usable sqlite3 CLI. macOS ships /usr/bin/sqlite3; Linux usually has it on PATH."""
    for candidate in ("sqlite3", "/usr/bin/sqlite3", "/opt/homebrew/bin/sqlite3"):
        try:
            res = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _recover_via_cli(src: Path, dst: Path, sqlite3_bin: str) -> tuple[bool, str]:
    """Use `sqlite3 .recover` to produce a clean DB. Returns (ok, message)."""
    try:
        # `.recover` writes SQL to stdout that reconstructs everything readable.
        # Pipe directly into a fresh DB.
        recovery_sql = subprocess.run(
            [sqlite3_bin, str(src), ".recover"],
            capture_output=True, timeout=300,
        )
        if recovery_sql.returncode != 0:
            return False, f".recover exited {recovery_sql.returncode}: {recovery_sql.stderr.decode()[:300]}"
        if not recovery_sql.stdout:
            return False, ".recover produced no output"

        load = subprocess.run(
            [sqlite3_bin, str(dst)],
            input=recovery_sql.stdout,
            capture_output=True, timeout=300,
        )
        if load.returncode != 0:
            return False, f"loading recovered SQL failed: {load.stderr.decode()[:300]}"
        return True, f"recovered via sqlite3 .recover ({len(recovery_sql.stdout):,} bytes of SQL)"
    except subprocess.TimeoutExpired:
        return False, ".recover timed out"


def _recover_via_python(src: Path, dst: Path) -> tuple[bool, str]:
    """Fallback row-walk recovery using the stdlib sqlite3 module."""
    try:
        src_conn = sqlite3.connect(str(src))
        src_cur = src_conn.cursor()
        src_cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
        create_stmts = [r[0] for r in src_cur.fetchall()]
        src_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
        table_names = [r[0] for r in src_cur.fetchall()]

        dst_conn = sqlite3.connect(str(dst))
        dst_cur = dst_conn.cursor()
        for s in create_stmts:
            try:
                dst_cur.execute(s)
            except sqlite3.OperationalError as e:
                print(f"  warning: could not recreate table from `{s[:80]}...`: {e}")
        dst_conn.commit()

        total_recovered = 0
        total_failed = 0
        for tbl in table_names:
            try:
                src_cur.execute(f"SELECT * FROM {tbl}")
                cols = [d[0] for d in src_cur.description]
                placeholders = ",".join("?" * len(cols))
                insert_sql = f"INSERT OR IGNORE INTO {tbl} ({','.join(cols)}) VALUES ({placeholders})"
                n_recovered = 0
                n_failed = 0
                while True:
                    try:
                        row = src_cur.fetchone()
                    except sqlite3.DatabaseError:
                        n_failed += 1
                        break
                    if row is None:
                        break
                    try:
                        dst_cur.execute(insert_sql, row)
                        n_recovered += 1
                    except sqlite3.IntegrityError:
                        n_failed += 1
                dst_conn.commit()
                total_recovered += n_recovered
                total_failed += n_failed
                print(f"  {tbl}: recovered {n_recovered}, lost {n_failed}")
            except sqlite3.DatabaseError as e:
                print(f"  {tbl}: could not recover — {e}")

        src_conn.close()
        dst_conn.close()
        return True, f"recovered via Python row-walk: {total_recovered} row(s), lost {total_failed}"
    except Exception as e:
        return False, f"Python recovery raised: {e}"


def recover(db_path: Path) -> int:
    """Returns shell-style exit code: 0 on success, 1 on hard failure."""
    if not db_path.exists():
        print(f"  [no file] {db_path}")
        return 0

    print(f"Checking {db_path} ({db_path.stat().st_size:,} bytes)...")
    ok, issues = _integrity_check(db_path)
    if ok:
        print(f"  integrity_check: ok — nothing to recover")
        return 0
    print(f"  integrity_check: {len(issues)} issue(s)")
    for issue in issues[:5]:
        print(f"    - {issue[:160]}")
    if len(issues) > 5:
        print(f"    - ... ({len(issues) - 5} more)")
    print()

    candidate = db_path.with_suffix(db_path.suffix + ".recovered")
    if candidate.exists():
        candidate.unlink()

    sqlite3_bin = _which_sqlite3()
    if sqlite3_bin:
        print(f"Recovering via sqlite3 CLI ({sqlite3_bin})...")
        ok, msg = _recover_via_cli(db_path, candidate, sqlite3_bin)
        print(f"  {msg}")
        if not ok:
            print(f"  CLI recovery failed; falling back to Python row-walk.")
            if candidate.exists():
                candidate.unlink()
            ok, msg = _recover_via_python(db_path, candidate)
            print(f"  {msg}")
    else:
        print("Recovering via Python row-walk (sqlite3 CLI not on PATH)...")
        ok, msg = _recover_via_python(db_path, candidate)
        print(f"  {msg}")

    if not ok or not candidate.exists():
        print("\nRecovery FAILED. Original file is untouched.")
        return 1

    rec_ok, rec_issues = _integrity_check(candidate)
    if not rec_ok:
        print(f"\nrecovered DB integrity: {len(rec_issues)} issue(s) remain")
        for issue in rec_issues[:5]:
            print(f"  - {issue[:160]}")
        print(
            "\nRefusing to promote a damaged recovered file. The original is "
            "untouched. Inspect manually, or fall back to a good backup."
        )
        return 1
    print(f"\n  recovered DB integrity: ok")

    quarantine = db_path.with_name(
        f"{db_path.name}.corrupt-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    shutil.move(str(db_path), str(quarantine))
    print(f"  quarantined corrupt file: {db_path} -> {quarantine}")
    shutil.move(str(candidate), str(db_path))
    print(f"  promoted recovered DB:    {candidate} -> {db_path}")

    # Report what we got
    conn = sqlite3.connect(str(db_path))
    try:
        per_thread = conn.execute(
            "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id ORDER BY thread_id"
        ).fetchall()
        n_writes = conn.execute("SELECT COUNT(*) FROM writes").fetchone()[0]
    finally:
        conn.close()
    print(f"\nRecovered {len(per_thread)} thread(s), {sum(c for _, c in per_thread)} checkpoint(s), {n_writes} write(s):")
    for tid, cnt in per_thread:
        print(f"  {tid}: {cnt}")
    return 0


def main() -> int:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints.db")
    return recover(db_path)


if __name__ == "__main__":
    sys.exit(main())
