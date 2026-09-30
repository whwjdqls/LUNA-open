"""Inspect/download selected members of the official uncompressed LHM prior TAR.

Source: pinned LHM commit 4f88aaeb's download_weights.sh. HTTP byte ranges avoid
fetching its entire 18.8 GB bundle. ETag/length consistency is checked throughout;
selected files receive local SHA256 receipts. This is not a license grant.
"""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

import requests

URL = (
    "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/"
    "share/aigc3d/data/for_lingteng/LHM/LHM_prior_model.tar"
)


class RemoteTar(io.RawIOBase):
    def __init__(self, session, length, etag):
        super().__init__()
        self.session, self.length, self.etag = session, length, etag
        self.position, self.cache_start, self.cache = 0, 0, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, self.length)[whence] + offset
        if not 0 <= self.position <= self.length:
            raise ValueError("Seek outside archive")
        return self.position

    def request(self, start, end):
        response = self.session.get(
            URL,
            headers={"Range": f"bytes={start}-{end}", "If-Match": self.etag},
            stream=True,
            timeout=(15, 60),
        )
        expected = f"bytes {start}-{end}/{self.length}"
        if (
            response.status_code != 206
            or response.headers.get("Content-Range") != expected
            or response.headers.get("ETag") != self.etag
        ):
            response.close()
            raise ValueError("Server did not honor a consistent bounded range request")
        return response

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded archive reads are not allowed")
        size = min(size, self.length - self.position)
        if not size:
            return b""
        if not (
            self.cache_start <= self.position
            and self.position + size <= self.cache_start + len(self.cache)
        ):
            self.cache_start = self.position
            end = min(self.position + max(size, 65536), self.length) - 1
            with self.request(self.position, end) as response:
                self.cache = response.content
            if len(self.cache) != end - self.position + 1:
                raise ValueError("Incomplete archive range")
        offset = self.position - self.cache_start
        self.position += size
        return self.cache[offset : offset + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", nargs="*", default=[], help="Exact archive member names")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        response = session.head(URL, timeout=(15, 60), allow_redirects=True)
        response.raise_for_status()
        etag, length = response.headers["ETag"], int(response.headers["Content-Length"])
        source = dict(
            url=URL, etag=etag, bytes=length, last_modified=response.headers.get("Last-Modified")
        )
        stream = RemoteTar(session, length, etag)
        index_path = args.output / "archive-index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if index["source"] != source:
                raise ValueError("Upstream archive changed; preserve existing acquisition")
        else:
            members = {}
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                for member in archive:
                    if member.isfile():
                        if member.name in members:
                            raise ValueError("Duplicate archive member")
                        members[member.name] = dict(size=member.size, offset=member.offset_data)
            index = dict(source=source, members=members)
            index_path.write_text(json.dumps(index, indent=2) + "\n")
        print("Indexed", len(index["members"]), "files in", length, "archive bytes", flush=True)
        receipts_path = args.output / "files.json"
        receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
        for name in args.files:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Expected a relative member path")
            member = index["members"][name]
            destination = args.output / relative
            if destination.exists():
                raise FileExistsError(f"Preserve the existing file: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            digest, count = hashlib.sha256(), 0
            if member["size"] <= 0:
                raise ValueError("Selected weight/asset file is empty")
            with stream.request(
                member["offset"], member["offset"] + member["size"] - 1
            ) as transfer:
                with temporary.open("wb") as output:
                    for chunk in transfer.iter_content(1024 * 1024):
                        count += len(chunk)
                        digest.update(chunk)
                        output.write(chunk)
            if count != member["size"]:
                raise ValueError(f"Incomplete member transfer: {name}")
            temporary.replace(destination)
            receipts[name] = dict(source=source, bytes=count, sha256=digest.hexdigest())
            receipts_path.write_text(json.dumps(receipts, indent=2) + "\n")
            print("Downloaded", name, count, digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
