"""Delete only SDK-owned inputs after verified, warning-free requested outputs."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Mapping

from .commit_journal import file_sha256
from .sdk_state import JobRecord, RetentionPolicy


def _verified_outputs(outputs: Mapping[str, str]) -> bool:
    return bool(outputs) and all(Path(path).is_file() and file_sha256(Path(path)) == digest
                                 for path, digest in outputs.items())


def cleanup_owned_session(record: JobRecord, session: Path, *, policy: RetentionPolicy,
                          formal_2d_published: bool, three_d_requested: bool,
                          three_d_succeeded: bool, output_sha256: Mapping[str, str]) -> bool:
    if (policy != RetentionPolicy.DELETE_AFTER_ALL_SUCCESS or not record.payload["owns_session"]
            or not formal_2d_published or record.payload.get("warnings")
            or record.payload.get("failure") or (three_d_requested and not three_d_succeeded)
            or not _verified_outputs(output_sha256)):
        return False
    session = session.resolve()
    session.relative_to((record.root / "session").resolve())
    if not session.is_dir():
        return False
    pending = session.with_name(session.name + ".delete_pending")
    if pending.exists():
        raise FileExistsError(pending)
    # Persist the intended rename first so a crash immediately after rename
    # still has an explicit path and ownership record for the next process.
    record.update(cleanup={"state": "rename_pending", "original": str(session),
                           "pending": str(pending), "output_sha256": dict(output_sha256)})
    session.rename(pending)
    record.update(cleanup={**record.payload["cleanup"], "state": "delete_pending"})
    shutil.rmtree(pending)
    record.update(cleanup={**record.payload["cleanup"], "state": "completed"})
    return True


def resume_cleanup(record: JobRecord) -> bool:
    cleanup = record.payload.get("cleanup", {})
    if (cleanup.get("state") not in {"rename_pending", "delete_pending"}
            or not record.payload.get("owns_session")
            or any(not warning.startswith("cleanup_pending:") for warning in record.payload.get("warnings", []))
            or record.payload.get("failure") or not _verified_outputs(cleanup.get("output_sha256", {}))):
        return False
    original, pending = Path(cleanup["original"]).resolve(), Path(cleanup["pending"]).resolve()
    original.relative_to((record.root / "session").resolve())
    pending.relative_to((record.root / "session").resolve())
    if pending != original.with_name(original.name + ".delete_pending"):
        raise ValueError("Invalid cleanup tombstone")
    if original.exists() and not pending.exists():
        original.rename(pending)
    if pending.exists():
        shutil.rmtree(pending)
    record.update(cleanup={**cleanup, "state": "completed"})
    return True
