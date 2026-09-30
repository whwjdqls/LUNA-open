"""Inventory/download an authorized DNA-Rendering Drive release into private storage.

Supply release URLs locally; do not commit URLs, file IDs, credentials, manifests,
or data. Uses Drive's ordinary large-file confirmation, never a quota bypass.
Folder HTML is not a stable API: ambiguous or potentially paginated listings fail.
Run bulk transfers and SHA256 checks on a Slurm CPU allocation.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def safe_name(name):
    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Unsafe release filename")
    return name


def folder_rows(html):
    match = re.search(r"window\['_DRIVE_ivd'\] = '(.*?)';", html, re.S)
    if not match:
        raise ValueError("Drive folder metadata missing; authentication or page format changed")
    escaped = re.sub(r"\\x([0-9A-Fa-f]{2})", r"\\u00\1", match[1]).replace("\\'", "'")
    rows = json.loads(json.loads('"' + escaped + '"'))[0]
    if not isinstance(rows, list) or len(rows) >= 50:
        raise ValueError("Potentially paginated folder: refusing to claim a complete inventory")
    entries = []
    for row in rows:
        entry = dict(id=row[0], name=safe_name(row[2]), mime=row[3], modified_ms=row[10])
        if entry["mime"] != "application/vnd.google-apps.folder":
            if not isinstance(row[13], int) or row[13] <= 0:
                raise ValueError("Missing exact Drive file size")
            entry["bytes"] = row[13]
        entries.append(entry)
    return entries


def folder_id(url):
    parsed = urlparse(url)
    match = re.fullmatch(r"/drive/folders/([A-Za-z0-9_-]+)", parsed.path)
    if parsed.scheme != "https" or parsed.netloc != "drive.google.com" or not match:
        raise ValueError("Expected an HTTPS Google Drive folder URL")
    return match[1]


def inventory(args, session):
    visited, files = set(), []
    listing_dir = args.root / "release" / "listings"
    listing_dir.mkdir(parents=True, exist_ok=True)

    def listing(fid):
        response = session.get(f"https://drive.google.com/drive/folders/{fid}", timeout=(15, 60))
        response.raise_for_status()
        (listing_dir / f"{fid}.html").write_text(response.text)
        return folder_rows(response.text)

    def walk(fid, prefix):
        if fid in visited:
            raise ValueError("Repeated/cyclic folder in release")
        visited.add(fid)
        for entry in listing(fid):
            relative = prefix / entry.pop("name")
            if entry["mime"] == "application/vnd.google-apps.folder":
                walk(entry["id"], relative)
            else:
                files.append(dict(path=relative.as_posix(), **entry))

    walk(folder_id(args.folder_url), Path("part1"))
    if args.support_folder_url:
        for entry in listing(folder_id(args.support_folder_url)):
            if entry["name"] in {"ReadMe.txt", "dna_rendering_sample_code.zip"}:
                name = entry.pop("name")
                files.append(dict(path=f"release/{name}", **entry))
    if len({entry["path"] for entry in files}) != len(files):
        raise ValueError("Duplicate release path")
    manifest = dict(
        schema=1,
        inspected_at=now(),
        source_folder=args.folder_url,
        bytes=sum(entry["bytes"] for entry in files),
        files=sorted(files, key=lambda item: item["path"]),
        restriction="Authorized research copy; no distribution or redistribution",
    )
    output = args.root / "manifest.json"
    if output.exists():
        old = json.loads(output.read_text())
        if old["files"] != manifest["files"] or old["source_folder"] != manifest["source_folder"]:
            raise ValueError("Release changed: preserve the previous manifest and use a new root")
    else:
        write_json(output, manifest)
    print(json.dumps(dict(files=len(files), bytes=manifest["bytes"], manifest=str(output))))


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.fields, self.inside = None, {}, False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action, self.inside = attrs.get("action"), True
        if tag == "input" and self.inside and attrs.get("type") == "hidden":
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.inside = False


class DriveQuotaError(RuntimeError):
    pass


def binary_response(session, fid, headers):
    url, params = "https://drive.google.com/uc", dict(export="download", id=fid)
    for _ in range(2):
        response = session.get(url, params=params, headers=headers, stream=True, timeout=(15, 90))
        if response.status_code == 429:
            response.close()
            raise DriveQuotaError("Drive HTTP 429; no automatic retry")
        if "text/html" not in response.headers.get("Content-Type", ""):
            response.raise_for_status()
            return response
        html = response.text
        response.close()
        if "Quota exceeded" in html or "Too many users have viewed or downloaded" in html:
            raise DriveQuotaError("Google Drive file download quota exceeded; no automatic retry")
        form = DownloadForm()
        form.feed(html)
        if (
            form.action != "https://drive.usercontent.google.com/download"
            or form.fields.get("id") != fid
        ):
            raise RuntimeError("Drive returned an HTML access/error page, not dataset bytes")
        url, params = form.action, form.fields
    raise RuntimeError("Drive did not return a binary after its normal download confirmation")


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest


def download_file(root, entry, session, previous, restart_partials):
    path = root / entry["path"]
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes data root")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != entry["bytes"]:
            raise ValueError("Existing final file has the wrong size; preserved")
        digest = digest_file(path).hexdigest()
        if previous.get("sha256") and digest != previous["sha256"]:
            raise ValueError("Existing final file differs from its SHA256 receipt; preserved")
        return dict(status="complete", bytes=entry["bytes"], sha256=digest, verified_at=now())
    partial = path.with_name(path.name + ".part")
    state_path = path.with_name(path.name + ".part.json")
    offset, headers = 0, {"Accept-Encoding": "identity"}
    if partial.exists() and partial.stat().st_size and not restart_partials:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("entry") != entry or not state.get("etag"):
            raise ValueError(
                "Partial lacks a matching strong ETag; use --restart-partials to restart"
            )
        offset = partial.stat().st_size
        if offset >= entry["bytes"]:
            raise ValueError("Partial size invalid for resume; use --restart-partials")
        headers.update(Range=f"bytes={offset}-", **{"If-Range": state["etag"]})
    with binary_response(session, entry["id"], headers) as response:
        if offset:
            expected = f"bytes {offset}-{entry['bytes'] - 1}/{entry['bytes']}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Server did not honor resume; partial preserved")
            if response.headers.get("ETag") != state["etag"]:
                raise ValueError("Resume ETag changed; partial preserved")
        elif response.status_code != 200:
            raise ValueError("Expected a full HTTP 200 download")
        length = response.headers.get("Content-Length")
        if length and int(length) != entry["bytes"] - offset:
            raise ValueError("HTTP size differs from the frozen Drive inventory")
        etag = response.headers.get("ETag")
        write_json(
            state_path, dict(entry=entry, etag=etag if etag and not etag.startswith("W/") else None)
        )
        digest = digest_file(partial) if offset else hashlib.sha256()
        count = offset
        with partial.open("ab" if offset else "wb") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                count += len(chunk)
                if count > entry["bytes"]:
                    raise ValueError("Download exceeded expected size")
                handle.write(chunk)
                digest.update(chunk)
        if count != entry["bytes"]:
            raise ValueError("Incomplete transfer; partial preserved")
    with partial.open("rb") as handle:
        magic = handle.read(8)
    if path.suffix == ".smc" and magic != b"\x89HDF\r\n\x1a\n":
        raise ValueError("SMC download does not have an HDF5 signature")
    if path.suffix == ".zip" and not magic.startswith(b"PK\x03\x04"):
        raise ValueError("ZIP download does not have a ZIP signature")
    if path.suffix == ".json":
        json.loads(partial.read_text())
    partial.replace(path)
    state_path.unlink()
    return dict(status="complete", bytes=count, sha256=digest.hexdigest(), verified_at=now())


def download(args, session):
    manifest_path = args.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = [item for item in manifest["files"] if not args.files or item["path"] in args.files]
    if args.files and set(args.files) != {item["path"] for item in selected}:
        raise ValueError("Requested path is absent from inventory")
    remaining = sum(item["bytes"] for item in selected if not (args.root / item["path"]).exists())
    if shutil.disk_usage(args.root).free < remaining + 20 * 1024**3:
        raise ValueError("Insufficient free storage for selected files plus a 20 GiB reserve")
    receipt_path = args.root / "acquisition.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"files": {}}
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if receipt.get("manifest_sha256", manifest_hash) != manifest_hash:
        raise ValueError("Receipt refers to a different manifest")
    receipt["manifest_sha256"] = manifest_hash
    failures = quota_errors = 0
    for entry in selected:
        name = entry["path"]
        try:
            result = download_file(
                args.root, entry, session, receipt["files"].get(name, {}), args.restart_partials
            )
        except (requests.RequestException, RuntimeError, ValueError, OSError) as error:
            failures += 1
            quota_errors += isinstance(error, DriveQuotaError)
            result = dict(
                status="quota_exceeded" if isinstance(error, DriveQuotaError) else "error",
                error=str(error),
                attempted_at=now(),
            )
        # Preserve an earlier hash even if re-verification fails; a later retry
        # must not silently accept the changed file as a new trusted baseline.
        receipt["files"][name] = {**receipt["files"].get(name, {}), **result}
        receipt["updated_at"] = now()
        write_json(receipt_path, receipt)
        print(json.dumps(dict(path=name, **result)), flush=True)
        if quota_errors >= args.max_quota_errors:
            print(
                "Stopping on repeated Drive quota errors; remaining files are unattempted",
                flush=True,
            )
            break
    complete = sum(item.get("status") == "complete" for item in receipt["files"].values())
    print(json.dumps(dict(complete=complete, total=len(manifest["files"]), failures=failures)))
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "download"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--folder-url")
    parser.add_argument("--support-folder-url")
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--restart-partials", action="store_true")
    parser.add_argument("--max-quota-errors", type=int, default=3)
    args = parser.parse_args()
    if args.max_quota_errors < 1:
        parser.error("--max-quota-errors must be positive")
    if args.root.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Dataset storage must be outside the source repository")
    if args.action == "inventory" and not args.folder_url:
        parser.error("inventory requires --folder-url")
    os.umask(0o077)
    args.root.mkdir(parents=True, exist_ok=True)
    args.root.chmod(0o700)
    with (args.root / ".acquire.lock").open("a") as lock, requests.Session() as session:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session.headers["User-Agent"] = "Mozilla/5.0"
        if args.action == "inventory":
            inventory(args, session)
        else:
            return download(args, session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
