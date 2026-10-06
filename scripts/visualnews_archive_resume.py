"""Validated HTTP ranges and logical tar boundaries; no extracted image files."""

import errno
import hashlib
import json
import os
import re
import socket
import tarfile
import uuid
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit


class ArchiveResumeError(RuntimeError):
    """Transport/format/state errors are systemic, not image decode failures."""


def sync_handle(handle):
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
            raise


def sync_directory(path):
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
            raise


@contextmanager
def writer_lock(state_path):
    """A hard-disconnect lock needs manual review, never automatic eviction."""
    lock_path = state_path.with_name(state_path.name + ".lock")
    token = uuid.uuid4().hex
    try:
        with lock_path.open("x") as handle:
            json.dump({"token": token, "pid": os.getpid(), "host": socket.gethostname()}, handle)
            sync_handle(handle)
    except FileExistsError as error:
        raise ArchiveResumeError(f"STOP: writer lock exists: {lock_path}. Confirm the old writer is stopped before removing this lock only.") from error
    try:
        yield
    finally:
        if lock_path.exists() and json.loads(lock_path.read_text()).get("token") == token:
            lock_path.unlink()
            sync_directory(lock_path.parent)


def prefix_sha256(path, length):
    digest = hashlib.sha256()
    if length == 0:
        return digest.hexdigest()
    if not path.is_file() or path.stat().st_size < length:
        raise ArchiveResumeError("STOP: cursor failure-journal prefix is missing; preserve state and repair it")
    with path.open("rb") as handle:
        remaining = length
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                raise ArchiveResumeError("STOP: truncated failure journal")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


class CheckedRangeReader:
    """Bound reads to Content-Range; premature EOF is always a transport error."""

    def __init__(self, raw, length):
        self.raw = raw
        self.remaining = length

    def read(self, size):
        if size is None or size < 0:
            raise ArchiveResumeError("Unbounded HTTP reads are not allowed")
        wanted = min(size, self.remaining)
        parts = []
        while wanted:
            try:
                block = self.raw.read(wanted)
            except Exception as error:
                raise ArchiveResumeError("STOP: HTTP range interrupted; resume from the committed cursor") from error
            if not block or len(block) > wanted:
                raise ArchiveResumeError("STOP: truncated or oversized HTTP range body")
            parts.append(block)
            self.remaining -= len(block)
            wanted -= len(block)
        return b"".join(parts)


def conditional_headers(identity):
    if identity is None:
        return {}
    etag = identity.get("etag")
    if etag and not etag.startswith("W/"):
        return {"If-Range": etag, "If-Match": etag}
    if identity.get("last_modified_is_strong"):
        return {"If-Unmodified-Since": identity["last_modified"]}
    raise ArchiveResumeError("STOP: source has no usable strong ETag/date validator; cannot safely resume bytes")


def response_identity(response, url, total):
    etag = response.headers.get("ETag")
    modified = response.headers.get("Last-Modified")
    strong_date = False
    if modified and response.headers.get("Date"):
        try:
            age = parsedate_to_datetime(response.headers["Date"]) - parsedate_to_datetime(modified)
            strong_date = age.total_seconds() >= 60
        except (TypeError, ValueError, OverflowError):
            pass
    identity = {"url": url, "resolved_url": getattr(response, "url", url), "size": total,
                "etag": etag, "last_modified": modified, "last_modified_is_strong": strong_date,
                "format": "uncompressed_tar_512"}
    if etag and not etag.startswith("W/") and not (etag.startswith('"') and etag.endswith('"')):
        raise ArchiveResumeError("STOP: malformed source ETag")
    conditional_headers(identity)
    return identity


def validate_range_response(response, url, start, end, expected_identity):
    if response.status_code != 206:
        raise ArchiveResumeError(f"STOP: range request returned HTTP {response.status_code}, not 206; no archive body accepted")
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ArchiveResumeError("STOP: encoded HTTP body is incompatible with absolute tar offsets")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
    if not match:
        raise ArchiveResumeError("STOP: malformed/missing Content-Range")
    first, last, total = map(int, match.groups())
    expected_end = total - 1 if end is None else end
    if first != start or last != expected_end or last < first or last >= total or total % 512:
        raise ArchiveResumeError("STOP: incorrect Content-Range or non-block-aligned source length")
    length = last - first + 1
    content_length = response.headers.get("Content-Length")
    if content_length is not None and content_length != str(length):
        raise ArchiveResumeError("STOP: Content-Length disagrees with requested range")
    identity = response_identity(response, url, total)
    if expected_identity is not None:
        for key in ["url", "resolved_url", "size", "etag", "last_modified", "format"]:
            if expected_identity.get(key) is not None and identity.get(key) != expected_identity[key]:
                raise ArchiveResumeError("STOP: archive source identity changed/missing: " + key + "; keep existing checkpoints and reconcile the source")
        identity = expected_identity.copy()
    return identity, length


class HTTPRangeSource:
    """Each response is validated before reading bytes; ranges never auto-fall back."""

    def __init__(self, url, request_get=None):
        if urlsplit(url).scheme not in ("http", "https"):
            raise ValueError("Archive URL must be HTTP(S)")
        if request_get is None:
            import requests
            request_get = requests.get
        self.url = url
        self.request_get = request_get

    @contextmanager
    def open_range(self, start, end=None, identity=None):
        if start < 0 or (end is not None and end < start):
            raise ValueError("Invalid byte range")
        headers = {"Range": f"bytes={start}-" + ("" if end is None else str(end)),
                   "Accept-Encoding": "identity"}
        headers.update(conditional_headers(identity))
        try:
            response = self.request_get(self.url, headers=headers, stream=True, timeout=(10, 60))
        except Exception as error:
            raise ArchiveResumeError("STOP: archive HTTP request failed; preserve state and retry") from error
        try:
            actual_identity, length = validate_range_response(response, self.url, start, end, identity)
            response.raw.decode_content = False
            yield CheckedRangeReader(response.raw, length), actual_identity
        finally:
            response.close()

    def probe(self, identity=None):
        with self.open_range(0, 511, identity) as (reader, actual_identity):
            header = reader.read(512)
            if header.startswith((b"\x1f\x8b", b"BZh", b"\xfd7zXZ")):
                raise ArchiveResumeError("STOP: compressed tar is unsupported for byte-position resume")
            try:
                tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
            except tarfile.HeaderError as error:
                raise ArchiveResumeError("STOP: source does not begin with a valid uncompressed tar header") from error
        return actual_identity

    @contextmanager
    def open_tar(self, offset, pax_headers, identity):
        if offset % 512:
            raise ArchiveResumeError("STOP: archive cursor is not a tar block boundary")
        with self.open_range(offset, identity=identity) as (reader, _):
            try:
                with tarfile.open(fileobj=reader, mode="r|", format=tarfile.PAX_FORMAT,
                                  tarinfo=StrictTarInfo, pax_headers=pax_headers.copy()) as archive:
                    yield archive
            except (tarfile.TarError, UnicodeError, OverflowError) as error:
                raise ArchiveResumeError("STOP: invalid/truncated or unsupported tar; cursor has not advanced past uncommitted work") from error

    def read_image_bytes(self, data_offset, size, identity):
        if size <= 0 or size > 32 * 1024 * 1024 or data_offset < 0 or data_offset + size > identity["size"]:
            raise ValueError("Failed target is not a bounded regular image payload")
        with self.open_range(data_offset, data_offset + size - 1, identity) as (reader, _):
            return reader.read(size)


def check_pax_headers(headers, global_header=False):
    if len(json.dumps(headers)) > 64 * 1024:
        raise ArchiveResumeError("STOP: oversized PAX state")
    if any(key.startswith("GNU.sparse") or key == "SCHILY.realsize" for key in headers):
        raise ArchiveResumeError("STOP: sparse tar members are unsupported")
    if global_header and "size" in headers:
        raise ArchiveResumeError("STOP: global PAX size overrides are unsupported; no unsafe cursor written")


class StrictTarInfo(tarfile.TarInfo):
    """Keep Python's GNU/PAX parsing, but never treat corrupt headers as normal EOF."""

    @classmethod
    def fromtarfile(cls, archive):
        return cls._fromtarfile(archive)

    @classmethod
    def _fromtarfile(cls, archive, dircheck=True):
        header = archive.fileobj.read(512)
        if header == bytes(512):
            if archive.fileobj.read(512) != bytes(512):
                raise ArchiveResumeError("STOP: tar requires two complete zero end blocks")
            raise tarfile.EOFHeaderError("end of tar")
        if len(header) != 512:
            raise ArchiveResumeError("STOP: truncated tar header")
        try:
            member = cls.frombuf(header, archive.encoding, archive.errors)
        except tarfile.HeaderError as error:
            raise ArchiveResumeError("STOP: invalid tar header checksum/structure") from error
        member.offset = archive.fileobj.tell() - 512  # Logical parser position, not HTTP physical position.
        return member._proc_member(archive)

    def _proc_member(self, archive):
        if self.type == tarfile.GNUTYPE_SPARSE:
            raise ArchiveResumeError("STOP: GNU sparse tar is unsupported")
        extensions = (tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK,
                      tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.SOLARIS_XHDTYPE)
        if self.type in extensions:
            if self.size < 0 or self.size > 1024 * 1024:
                raise ArchiveResumeError("STOP: unsupported extension-header size")
        elif self.type not in tarfile.SUPPORTED_TYPES or self.size < 0 or (not self.isreg() and self.size):
            raise ArchiveResumeError("STOP: unsupported tar member type/size")
        check_pax_headers(archive.pax_headers, global_header=True)
        member = super()._proc_member(archive)
        check_pax_headers(member.pax_headers)
        check_pax_headers(archive.pax_headers, global_header=True)
        if member.size < 0 or (not member.isreg() and member.size):
            raise ArchiveResumeError("STOP: unsupported extended member size/type")
        if member.sparse is not None:
            raise ArchiveResumeError("STOP: sparse members cannot establish a resume boundary")
        return member

    def _proc_pax(self, archive):
        # Validate framing independently of the runtime's tarfile patch level.
        # The wrapper observes the parser's extension read without changing position.
        original = archive.fileobj
        declared_size = self.size
        class ValidatedPaxReader:
            first = True
            def read(inner, size):
                block = original.read(size)
                if inner.first:
                    inner.first = False
                    payload = block[:declared_size]
                    if len(payload) != declared_size:
                        raise ArchiveResumeError("STOP: truncated PAX extension")
                    position = 0
                    while position < declared_size:
                        match = re.match(rb"([1-9][0-9]*) ", payload[position:])
                        if not match:
                            raise ArchiveResumeError("STOP: malformed PAX record framing")
                        length = int(match[1])
                        end = position + length
                        if length < 5 or end > declared_size or payload[end - 1:end] != b"\n":
                            raise ArchiveResumeError("STOP: invalid PAX record length")
                        key, separator, value = payload[position + match.end():end - 1].partition(b"=")
                        if not key or not separator:
                            raise ArchiveResumeError("STOP: malformed PAX key/value")
                        if key == b"size" and (not value.isdigit() or len(value) > 20):
                            raise ArchiveResumeError("STOP: invalid PAX size override")
                        position = end
                return block
            def __getattr__(inner, name):
                return getattr(original, name)
        archive.fileobj = ValidatedPaxReader()
        try:
            return super()._proc_pax(archive)
        except tarfile.HeaderError as error:
            raise ArchiveResumeError("STOP: malformed PAX extension/header") from error
        finally:
            archive.fileobj = original

    def _proc_gnusparse_00(self, *args):
        raise ArchiveResumeError("STOP: PAX sparse tar is unsupported")

    _proc_gnusparse_01 = _proc_gnusparse_00
    _proc_gnusparse_10 = _proc_gnusparse_00


def member_boundary(archive, member, base_offset, archive_size):
    """Extensions belong to the member; next boundary includes payload padding."""
    next_offset = base_offset + member.offset_data
    if member.isreg():
        next_offset += ((member.size + 511) // 512) * 512
    if next_offset != base_offset + archive.offset or next_offset % 512 or next_offset > archive_size:
        raise ArchiveResumeError("STOP: parser boundary cannot be reconciled safely")
    return next_offset, archive.pax_headers.copy()


def state_checksum(record, field):
    body = {key: value for key, value in record.items() if key != field}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class ArchiveCursor:
    """Separate durable tar position from authoritative validated vector chunks."""

    def __init__(self, store):
        self.store = store
        self.path = store.state_path.with_name("image_archive_cursor.json")
        self.journal = store.state_path.with_name("image_archive_failures.jsonl")
        self.expected_hash = prefix_sha256(self.path, self.path.stat().st_size) if self.path.exists() else None
        self.journal_size = self.journal.stat().st_size if self.journal.exists() else 0
        self.journal_hash = prefix_sha256(self.journal, self.journal_size)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else None
        self.failures = {}
        if self.state:
            state = self.state
            if state.get("state_sha256") != state_checksum(state, "state_sha256"):
                raise ArchiveResumeError("STOP: archive cursor checksum mismatch; restore trusted cursor state")
            if state.get("schema_version") != 1 or state.get("signature") != store.signature:
                raise ArchiveResumeError("STOP: archive cursor schema/manifest/model mismatch")
            offset = state.get("next_offset")
            if type(offset) is not int or offset < 0 or offset % 512 or offset >= state["source"]["size"]:
                raise ArchiveResumeError("STOP: invalid committed tar offset")
            check_pax_headers(state["pax_headers"], global_header=True)
            if store.chunks[:len(state["checkpoints"])] != state["checkpoints"]:
                raise ArchiveResumeError("STOP: cursor references missing/changed checkpoint markers")
            if prefix_sha256(self.journal, state["failure_journal_bytes"]) != state["failure_journal_sha256"]:
                raise ArchiveResumeError("STOP: committed failure-journal checksum mismatch")
        if self.journal.exists():
            with self.journal.open() as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                        if record.get("record_sha256") != state_checksum(record, "record_sha256"):
                            raise ValueError("record checksum mismatch")
                        row = store.target_lookup.loc[record["metadata_id"]]
                        if record["signature"] != store.signature or record["image_path"] != row.image_path:
                            raise ValueError("scope/path mismatch")
                        if self.state and record["source"] != self.state["source"]:
                            raise ValueError("source mismatch")
                        if (record["member_offset"] < 0 or record["data_offset"] < record["member_offset"] + 512 or
                                record["member_offset"] % 512 or record["data_offset"] % 512 or record["size"] < 0 or
                                record["data_offset"] + record["size"] > record["source"]["size"]):
                            raise ValueError("invalid payload boundary")
                        self.failures[record["image_path"]] = record
                    except (KeyError, ValueError, TypeError) as error:
                        raise ArchiveResumeError("STOP: malformed/conflicting failure journal; preserve and repair it") from error

    def check_unchanged(self):
        actual = prefix_sha256(self.path, self.path.stat().st_size) if self.path.exists() else None
        if actual != self.expected_hash:
            raise ArchiveResumeError("STOP: archive cursor changed by another writer")
        size = self.journal.stat().st_size if self.journal.exists() else 0
        if size != self.journal_size or prefix_sha256(self.journal, size) != self.journal_hash:
            raise ArchiveResumeError("STOP: failure journal changed by another writer")

    def commit(self, offset, pax_headers, identity, eof=False):
        if self.store.pending_vectors or self.store.pending_rows:
            raise ArchiveResumeError("STOP: cannot commit cursor with pending vectors")
        if type(offset) is not int or offset % 512 or not 0 <= offset < identity["size"]:
            raise ArchiveResumeError("STOP: invalid candidate tar boundary")
        if self.state and offset < self.state["next_offset"]:
            raise ArchiveResumeError("STOP: cursor cannot move backwards")
        check_pax_headers(pax_headers, global_header=True)
        self.check_unchanged()
        self.store.check_state_unchanged()
        # Completion markers have already been published and validated by the store.
        if __package__:
            from .visualnews_stream_embeddings import write_json_atomic, utc_now
        else:
            from visualnews_stream_embeddings import write_json_atomic, utc_now
        state = {"schema_version": 1, "signature": self.store.signature, "source": identity,
                 "next_offset": offset, "pax_headers": pax_headers.copy(), "archive_eof": eof,
                 "checkpoints": self.store.chunks.copy(), "failure_journal_bytes": self.journal_size,
                 "failure_journal_sha256": self.journal_hash, "committed_at_utc": utc_now()}
        state["state_sha256"] = state_checksum(state, "state_sha256")
        write_json_atomic(self.path, state)
        self.state = state
        self.expected_hash = prefix_sha256(self.path, self.path.stat().st_size)

    def record_failure(self, row, error, member_offset, data_offset, size, regular, identity):
        self.check_unchanged()
        if __package__:
            from .visualnews_stream_embeddings import utc_now
        else:
            from visualnews_stream_embeddings import utc_now
        record = {"signature": self.store.signature, "source": identity,
                  "metadata_id": int(row["id"]), "image_path": row["image_path"],
                  "member_offset": member_offset, "data_offset": data_offset,
                  "size": size, "regular": regular, "error_type": type(error).__name__,
                  "error": str(error)[:1000], "timestamp_utc": utc_now()}
        record["record_sha256"] = state_checksum(record, "record_sha256")
        with self.journal.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            sync_handle(handle)
        sync_directory(self.journal.parent)
        self.journal_size = self.journal.stat().st_size
        self.journal_hash = prefix_sha256(self.journal, self.journal_size)
        self.failures[row["image_path"]] = record
        self.store.record_failure(row, error)  # Preserve the existing CSV schema too.
