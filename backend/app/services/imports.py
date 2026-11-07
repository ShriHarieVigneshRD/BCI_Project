"""Dataset import sessions: select a folder / files / local path -> scan -> review -> import.

* Browser uploads are planned from the file list first (only EEG recordings, event
  files, companions and metadata are transferred), then streamed to a staging folder
  in batches, preserving the relative folder structure (sanitised, never outside the
  staging folder).
* Local folder paths are read in place (never written to) after validation: absolute,
  existing, a directory, not a drive root, not a network share, not inside the app's
  data store, and inside ``EEGNET_IMPORT_ROOTS`` when that allow-list is configured.
* Scanning and importing run in background threads with progress reporting; signal
  data are read one recording at a time.
"""

from __future__ import annotations

import fnmatch
import json
import re
import shutil
import threading
import traceback
from pathlib import Path

from ..config import settings
from ..data.importer import import_manifest
from ..data.loaders import ALLOWED_EXTENSIONS
from ..data.manifest import ArrayOptions, DatasetManifest, FileEntry, validate_manifest
from ..data.scanner import MAX_FILES, classify_name, plan_files, scan_dataset
from ..data.store import load_dataset, summarize
from ..schemas import DatasetRecord, ImportSession, now_iso
from . import repo

_lock = threading.Lock()
_progress: dict[str, dict] = {}


class ImportRejected(ValueError):
    """Invalid request for an import session (HTTP 422)."""


class ImportConflict(RuntimeError):
    """Request conflicts with the session state (HTTP 409)."""


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _dir(sid: str) -> Path:
    p = settings.data_dir / "imports" / sid
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_session(s: ImportSession) -> ImportSession:
    s.updated_at = now_iso()
    repo.db().execute("INSERT OR REPLACE INTO import_sessions (id, created_at, status, data) VALUES (?, ?, ?, ?)",
                      (s.id, s.created_at, s.status, s.model_dump_json()))
    return s


def get_session(sid: str) -> ImportSession:
    rows = repo.db().query("SELECT data FROM import_sessions WHERE id = ?", (sid,))
    if not rows:
        raise repo.NotFound(f"Import session {sid} not found")
    s = ImportSession.model_validate_json(rows[0]["data"])
    with _lock:
        if sid in _progress:
            s.progress = dict(_progress[sid])
    return s


def load_manifest(sid: str) -> tuple[DatasetManifest | None, dict]:
    d = settings.data_dir / "imports" / sid
    if not (d / "manifest.json").exists():
        return None, {}
    m = DatasetManifest.model_validate_json((d / "manifest.json").read_text(encoding="utf-8"))
    events = json.loads((d / "events.json").read_text(encoding="utf-8")) if (d / "events.json").exists() else {}
    return m, {int(k): v for k, v in events.items()}


def save_manifest(sid: str, m: DatasetManifest, events: dict | None = None) -> None:
    d = _dir(sid)
    (d / "manifest.json").write_text(m.model_dump_json(), encoding="utf-8")
    if events is not None:
        (d / "events.json").write_text(json.dumps({str(k): v for k, v in events.items()}), encoding="utf-8")


def source_root(s: ImportSession) -> Path:
    if s.local_path:
        return Path(s.local_path)
    if s.source_dir:
        return settings.data_dir / s.source_dir
    return _dir(s.id) / "files"


def _set_progress(sid: str, done: int, total: int, message: str) -> None:
    with _lock:
        _progress[sid] = {"done": done, "total": total, "message": message}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def validate_local_path(raw: str) -> Path:
    """Safety checks for reading a dataset folder in place on this machine."""
    if not raw or "\x00" in raw:
        raise ImportRejected("Enter the full path of a dataset folder.")
    p = raw.strip().strip('"').strip("'")
    if p.startswith("\\\\") or p.startswith("//"):
        raise ImportRejected("Network (UNC) paths are not allowed: EEG data must stay on this machine.")
    path = Path(p).expanduser()
    if not path.is_absolute():
        raise ImportRejected("Enter an absolute path, e.g. D:\\data\\my_study or /home/me/data/my_study.")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise ImportRejected("That folder does not exist or cannot be accessed.")
    if not resolved.is_dir():
        raise ImportRejected("That path is a file; choose the folder that contains the dataset.")
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 2:
        raise ImportRejected("Choose a dataset folder, not the root of a drive.")
    store = settings.data_dir.resolve()
    if resolved == store or store in resolved.parents or resolved in store.parents:
        raise ImportRejected("That folder is (or contains) the app's own data store; choose the original dataset.")
    if settings.import_roots:
        roots = [Path(r).expanduser().resolve() for r in settings.import_roots]
        if not any(resolved == r or r in resolved.parents for r in roots):
            raise ImportRejected("That folder is outside the allowed import locations (EEGNET_IMPORT_ROOTS).")
    return resolved


def safe_relpath(p: str) -> str:
    """Sanitise a browser-supplied relative path; never allows escaping the staging folder."""
    parts = [x for x in p.replace("\\", "/").split("/") if x not in ("", ".")]
    if not parts or any(x == ".." for x in parts) or re.match(r"^[A-Za-z]:", parts[0]):
        raise ImportRejected(f"Invalid file path: {p!r}")
    clean = [re.sub(r"[^A-Za-z0-9._\- ()+=,@]", "_", x).strip()[:150] for x in parts]
    if any(not c or c in (".", "..") for c in clean):
        raise ImportRejected(f"Invalid file path: {p!r}")
    return "/".join(clean)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def create_session(mode: str, name: str | None = None, local_path: str | None = None,
                   attach_experiment_id: str | None = None) -> ImportSession:
    sid = repo.new_id("import")
    ts = now_iso()
    s = ImportSession(id=sid, created_at=ts, updated_at=ts, mode=mode, name=name or None,
                      attach_experiment_id=attach_experiment_id)
    if mode == "local_path":
        root = validate_local_path(local_path or "")
        s.local_path = str(root)
        s.root_label = root.name
    else:
        s.source_dir = f"imports/{sid}/files"
        (settings.data_dir / s.source_dir).mkdir(parents=True, exist_ok=True)
    return save_session(s)


def plan(sid: str, files: list[dict]) -> dict:
    s = get_session(sid)
    if s.mode == "local_path":
        raise ImportConflict("Local-folder imports are scanned in place; no upload plan is needed.")
    if len(files) > MAX_FILES:
        raise ImportRejected(f"More than {MAX_FILES} files selected; choose the dataset folder itself.")
    roots = {f["path"].replace("\\", "/").split("/")[0] for f in files if "/" in f["path"].replace("\\", "/")}
    if s.mode == "folder" and len(roots) == 1:
        s.root_label = next(iter(roots))
        save_session(s)
    entries = plan_files(files)
    counts: dict[str, int] = {}
    size: dict[str, int] = {}
    for e in entries:
        counts[e.role] = counts.get(e.role, 0) + 1
        size[e.role] = size.get(e.role, 0) + e.size
    upload = [e for e in entries if e.role != "ignored"]
    total = sum(e.size for e in upload)
    if total > settings.max_import_mb * 1024 * 1024:
        raise ImportRejected(f"The selected files total {total / 1024 ** 3:.1f} GB, above the "
                             f"{settings.max_import_mb / 1024:.0f} GB upload limit. Use 'Local folder path' "
                             "instead (no upload needed).")
    return {"files": [e.model_dump() for e in entries], "counts": counts, "bytes": size, "upload_bytes": total,
            "upload_files": len(upload), "root_label": s.root_label}


def strip_root(path: str, root_label: str | None) -> str:
    p = path.replace("\\", "/")
    if root_label and p.startswith(root_label + "/"):
        return p[len(root_label) + 1:]
    return p


async def stage_files(sid: str, paths: list[str], uploads) -> dict:
    s = get_session(sid)
    if s.mode == "local_path" or s.status not in ("created", "uploading"):
        raise ImportConflict(f"Files cannot be added to a session that is '{s.status}'.")
    if len(paths) != len(uploads):
        raise ImportRejected("Each uploaded file needs a matching relative path.")
    base = (settings.data_dir / s.source_dir).resolve()
    limit = settings.max_import_mb * 1024 * 1024
    saved, skipped = [], []
    for rel_raw, up in zip(paths, uploads):
        rel = safe_relpath(strip_root(rel_raw, s.root_label if s.mode == "folder" else None))
        role, _ = classify_name(rel)
        ext = Path(rel.lower()).suffix
        if role == "ignored" or (ext not in ALLOWED_EXTENSIONS and ext not in (".json", ".md", "") and role != "metadata"):
            skipped.append(rel)
            continue
        target = (base / rel).resolve()
        if base not in target.parents:
            raise ImportRejected(f"Invalid file path: {rel_raw!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(target, "wb") as out:
            while True:
                chunk = await up.read(1 << 20)
                if not chunk:
                    break
                n += len(chunk)
                if s.staged_bytes + n > limit:
                    out.close()
                    target.unlink(missing_ok=True)
                    raise ImportRejected(f"Upload exceeds the {settings.max_import_mb / 1024:.0f} GB limit.")
                out.write(chunk)
        s.staged_bytes += n
        s.staged_files += 1
        saved.append(rel)
    s.status = "uploading"
    save_session(s)
    return {"saved": len(saved), "skipped": skipped, "staged_files": s.staged_files, "staged_bytes": s.staged_bytes}


def _run_bg(sid: str, fn, wait: bool) -> None:
    def runner():
        try:
            fn()
        except Exception as exc:
            traceback.print_exc()
            s = get_session(sid)
            s.status, s.error = "failed", f"{type(exc).__name__}: {exc}"
            save_session(s)
    if wait:
        runner()
    else:
        threading.Thread(target=runner, name=f"import-{sid}", daemon=True).start()


def start_scan(sid: str, array_options: ArrayOptions | None = None, wait: bool = False) -> ImportSession:
    s = get_session(sid)
    if s.status in ("scanning", "importing", "done"):
        raise ImportConflict(f"The session is '{s.status}'.")
    root = source_root(s)
    if s.mode != "local_path" and s.staged_files == 0:
        raise ImportConflict("No files have been uploaded yet.")
    s.status, s.error = "scanning", None
    save_session(s)
    _set_progress(sid, 0, 0, "Listing files")

    def job():
        source = {"mode": s.mode, "root_label": s.root_label, "import_id": sid}
        if s.local_path:
            source["local_path"] = s.local_path
        m, events = scan_dataset(root, s.name or "", source, array_options=array_options,
                                 progress=lambda d, t, msg: _set_progress(sid, d, t, msg))
        save_manifest(sid, m, events)
        cur = get_session(sid)
        cur.status = "review"
        save_session(cur)
        _set_progress(sid, 1, 1, "Ready for review")

    _run_bg(sid, job, wait)
    return get_session(sid)


def update_manifest(sid: str, patch: dict) -> DatasetManifest:
    s = get_session(sid)
    if s.status != "review":
        raise ImportConflict(f"The dataset can only be edited during review (session is '{s.status}').")
    m, events = load_manifest(sid)
    if m is None:
        raise ImportConflict("The session has not been scanned yet.")
    if "name" in patch and patch["name"]:
        m.name = str(patch["name"])[:200]
    by_idx = {r.index: r for r in m.recordings}
    for edit in patch.get("recordings") or []:
        r = by_idx.get(int(edit["index"]))
        if r is None:
            continue
        for key in ("participant", "session", "run", "task"):
            if key in edit:
                v = edit[key]
                v = re.sub(r"[^A-Za-z0-9_\-]", "", str(v))[:32] if v not in (None, "") else None
                setattr(r, key, v)
                if key == "participant":
                    r.id_source = "edited"
        if "included" in edit:
            r.included = bool(edit["included"]) and r.readable
            r.exclude_reason = None if r.included else (r.exclude_reason or "excluded by user")
    if patch.get("mapping") is not None:
        table = {(e.get("task"), e["event"]): e for e in patch["mapping"]}
        for e in m.mapping:
            new = table.get((e.task, e.event))
            if new:
                e.action = new.get("action", e.action)
                e.label = (str(new["label"]).strip()[:64] or None) if new.get("label") is not None else None
    for rule in patch.get("rules") or []:
        pat = str(rule.get("pattern", "")).strip()
        if not pat:
            continue
        for e in m.mapping:
            if (rule.get("task") in (None, "", e.task)) and fnmatch.fnmatch(e.event.lower(), pat.lower()):
                e.action = rule.get("action", "class")
                e.label = (str(rule.get("label")).strip() or None) if rule.get("label") else None
    for key in ("channel_policy", "sfreq_policy"):
        if patch.get(key):
            setattr(m, key, patch[key])
    if "target_sfreq" in patch:
        m.target_sfreq = float(patch["target_sfreq"]) if patch["target_sfreq"] else None
    rescan_arrays = False
    if patch.get("array_options") is not None:
        new_opts = ArrayOptions(**patch["array_options"])
        rescan_arrays = new_opts.ch_names != m.array_options.ch_names or new_opts.unit != m.array_options.unit
        m.array_options = new_opts
    if rescan_arrays:
        # channel names / unit of NumPy & CSV files change their header information: rescan those files only
        root = source_root(s)
        fresh, fresh_events = scan_dataset(root, m.name, m.source, files=[f.model_copy() for f in m.files],
                                           array_options=m.array_options)
        for r in m.recordings:
            if r.format in ("npz", "npy", "csv", "tsv"):
                fr = fresh.recordings[r.index]
                r.ch_names, r.ch_types, r.n_channels, r.unit = fr.ch_names, fr.ch_types, fr.n_channels, fr.unit
                events[r.index] = fresh_events.get(r.index, events.get(r.index))
    m = validate_manifest(m, events)
    save_manifest(sid, m, events)
    return m


def start_import(sid: str, wait: bool = False) -> ImportSession:
    s = get_session(sid)
    if s.status != "review":
        raise ImportConflict(f"Only a reviewed session can be imported (session is '{s.status}').")
    m, events = load_manifest(sid)
    m = validate_manifest(m, events)
    if not m.importable:
        errs = [i.message for i in m.issues if i.level == "error"]
        errs += [f"{r.file}: {i.message}" for r in m.recordings if r.included for i in r.issues if i.level == "error"]
        raise ImportRejected("Resolve these problems before importing: " + " · ".join(errs[:6]))
    s.status = "importing"
    save_session(s)
    _set_progress(sid, 0, m.summary["n_included"], "Starting import")

    def job():
        ds_id = repo.new_id("ds")
        out = settings.data_dir / "datasets" / ds_id
        fp = import_manifest(m, events, source_root(s), out, progress=lambda d, t, msg: _set_progress(sid, d, t, msg))
        ds = load_dataset(out)
        summary = summarize(ds)
        files = [f.path for f in m.files if f.role != "ignored" and f.recording is not None
                 and m.recordings[f.recording].included]
        src = {"mode": s.mode, "import_id": sid, "root_label": s.root_label, "revises": s.revises}
        if s.local_path:
            src["local_path"] = s.local_path
        else:
            src["source_dir"] = s.source_dir
        issues = [i.model_dump() for i in m.issues]
        rec = DatasetRecord(id=ds_id, name=m.name, created_at=now_iso(), format=_format_label(m),
                            structure=summary["structure"], fingerprint=fp, synthetic=False, files=files,
                            summary=summary, issues=issues, trainable=summary["n_classes"] >= 2,
                            notes=[f"Imported from {('local folder ' + s.local_path) if s.local_path else (s.root_label or 'uploaded files')} "
                                   f"({m.summary['n_included']} recordings, {m.summary['n_participants']} participants)."],
                            manifest=m.model_dump(mode="json"), source=src)
        repo.save_dataset(rec)
        cur = get_session(sid)
        cur.status, cur.dataset_id = "done", ds_id
        save_session(cur)
        _set_progress(sid, m.summary["n_included"], m.summary["n_included"], "Imported")

    _run_bg(sid, job, wait)
    return get_session(sid)


def _format_label(m: DatasetManifest) -> str:
    fmts = sorted({r.format for r in m.recordings if r.included})
    return fmts[0] if len(fmts) == 1 else "mixed: " + "+".join(fmts)


def cancel(sid: str) -> None:
    s = get_session(sid)
    if s.status in ("scanning", "importing"):
        raise ImportConflict("Wait for the current step to finish.")
    if s.status != "done":
        s.status = "cancelled"
        save_session(s)
        if s.source_dir and s.mode != "revise":
            shutil.rmtree(settings.data_dir / s.source_dir, ignore_errors=True)


def revise(dataset_id: str) -> ImportSession:
    """Start a new review from an imported dataset's manifest; importing creates a *new* dataset."""
    rec = repo.get_dataset(dataset_id)
    if not rec.manifest or rec.source.get("mode") not in ("folder", "files", "local_path", "revise"):
        raise ImportConflict("Only datasets imported through the dataset import workflow can be revised.")
    old_sid = rec.source.get("import_id")
    m, events = load_manifest(old_sid) if old_sid else (None, {})
    if m is None:
        raise ImportConflict("The original scan of this dataset is no longer available; import it again.")
    sid = repo.new_id("import")
    ts = now_iso()
    s = ImportSession(id=sid, created_at=ts, updated_at=ts, mode="revise", status="review", name=rec.name,
                      root_label=rec.source.get("root_label"), local_path=rec.source.get("local_path"),
                      source_dir=rec.source.get("source_dir"), revises=dataset_id)
    if s.local_path:
        validate_local_path(s.local_path)
    elif not s.source_dir or not (settings.data_dir / s.source_dir).exists():
        raise ImportConflict("The uploaded source files of this dataset are no longer available.")
    m = DatasetManifest.model_validate(rec.manifest)
    m.name = f"{rec.name} (revised)"
    save_manifest(sid, validate_manifest(m, events), events)
    return save_session(s)


def session_payload(sid: str, include_manifest: bool = True) -> dict:
    s = get_session(sid)
    out = s.model_dump(mode="json")
    if include_manifest:
        m, _ = load_manifest(sid)
        out["manifest"] = m.model_dump(mode="json") if m else None
    return out


def recover_interrupted() -> None:
    for row in repo.db().query("SELECT data FROM import_sessions WHERE status IN ('scanning', 'importing')"):
        s = ImportSession.model_validate_json(row["data"])
        s.status = "failed" if s.status == "importing" else "created"
        s.error = "Interrupted by a server restart; run the step again."
        save_session(s)


__all__ = ["ImportRejected", "ImportConflict", "FileEntry"]
