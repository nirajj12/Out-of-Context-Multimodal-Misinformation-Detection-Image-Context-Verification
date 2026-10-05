"""Extract only manifest-selected regular images from a user-supplied tar archive."""

import argparse
import os
import tarfile
import tempfile
from contextlib import contextmanager
from http.client import HTTPException
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

if __package__:
    from .check_missing_assets import (
        PROJECT_ROOT, default_image_directory, load_asset_manifest,
        safe_relative_path, staged_path,
    )
else:
    from check_missing_assets import (
        PROJECT_ROOT, default_image_directory, load_asset_manifest,
        safe_relative_path, staged_path,
    )


@contextmanager
def open_remote_response(url):
    try:
        import requests
    except ImportError:
        request = Request(url, headers={"Accept-Encoding": "identity"})
        try:
            with urlopen(request, timeout=60) as response:
                yield response, response.headers
        except HTTPException as error:
            raise OSError(f"Remote archive stream failed: {error}") from error
        return

    from urllib3.exceptions import HTTPError as StreamError

    try:
        with requests.get(
            url, stream=True, timeout=(10, 60),
            headers={"Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            yield response.raw, response.headers
    except (requests.RequestException, StreamError, HTTPException) as error:
        raise OSError(f"Remote archive stream failed: {error}") from error


@contextmanager
def open_archive(archive_path=None, archive_url=None):
    has_local_archive = archive_path is not None
    has_remote_archive = archive_url is not None
    if has_local_archive == has_remote_archive:
        raise ValueError("Provide exactly one local archive path or remote URL")
    if archive_url is not None:
        url_parts = urlsplit(archive_url)
        if url_parts.scheme not in ("http", "https") or not url_parts.hostname:
            raise ValueError("Archive URL must use HTTP or HTTPS and include a hostname")
        print(f"Streaming URL: {archive_url}", flush=True)
        with open_remote_response(archive_url) as (stream, headers):
            content_length = headers.get("Content-Length")
            if content_length is not None:
                print(f"Reported remote size: {content_length} bytes", flush=True)
                try:
                    content_length = int(content_length)
                except ValueError:
                    content_length = None
                if content_length is not None and content_length < 0:
                    content_length = None
            else:
                print("Remote size: Content-Length not provided", flush=True)
            with tarfile.open(fileobj=stream, mode="r|*") as archive:
                yield archive, content_length
    else:
        archive_path = Path(archive_path)
        if not archive_path.is_file() or not tarfile.is_tarfile(archive_path):
            raise ValueError(f"Not a readable tar archive: {archive_path}")
        print(f"Archive: {archive_path.resolve()}", flush=True)
        with tarfile.open(archive_path, mode="r|*") as archive:
            yield archive, None


def preview_archive(archive_path, archive_url, count):
    with open_archive(archive_path, archive_url) as (archive, _):
        shown = 0
        for member in archive:
            # repr keeps control characters in untrusted names out of the terminal.
            print(f"{shown + 1}: {member.name!r}")
            shown += 1
            if shown == count:
                break
    print(f"Previewed {shown} archive members; no files extracted.")


def requested_members(assets, strip_prefix="", member_prefix=""):
    strip_prefix = strip_prefix.rstrip("/")
    member_prefix = member_prefix.rstrip("/")
    if strip_prefix:
        safe_relative_path(strip_prefix)
    if member_prefix:
        safe_relative_path(member_prefix)
    members = {}
    for asset in assets.itertuples(index=False):
        archive_path = asset.normalized_archive_path
        if strip_prefix:
            prefix = strip_prefix + "/"
            if not archive_path.startswith(prefix):
                raise ValueError(f"Archive path does not start with {prefix!r}: {archive_path}")
            archive_path = archive_path[len(prefix):]
        if member_prefix:
            archive_path = member_prefix + "/" + archive_path
        safe_relative_path(archive_path)
        if archive_path in members:
            raise ValueError(f"Ambiguous requested archive member: {archive_path}")
        members[archive_path] = asset.image_path
    return members


def canonical_member_name(name):
    # Tar tools commonly add './'; no other path rewrites are implicit.
    while name.startswith("./"):
        name = name[2:]
    return str(safe_relative_path(name))


def copy_member(archive, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Cannot read selected member: {member.name}")
    temporary_path = None
    try:
        with source, tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".extract-", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            copied_bytes = 0
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                temporary.write(chunk)
                copied_bytes += len(chunk)
        if copied_bytes != member.size:
            raise ValueError(f"Incomplete selected member: {member.name}")
        # Publish a completed file without replacing a raced-in destination.
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def extract_selected(
    archive_path, assets, image_directory,
    strip_prefix="", member_prefix="", dry_run=False, progress_every=100,
    archive_url=None,
):
    requested = requested_members(assets, strip_prefix, member_prefix)
    found = set()
    matched = set()
    errors = []
    extracted_count = 0
    existing_count = 0
    unsafe_member_count = 0
    duplicate_member_count = 0
    scanned_count = 0

    print(f"Selected images: {len(requested)}; staging root: {Path(image_directory).resolve()}")
    with open_archive(archive_path, archive_url) as (archive, content_length):
        for member in archive:
            scanned_count += 1
            if scanned_count % 10_000 == 0:
                print(
                    f"Scanned {scanned_count:,} members; "
                    f"found {len(found):,} selected images",
                    flush=True,
                )
            try:
                member_name = canonical_member_name(member.name)
            except ValueError:
                unsafe_member_count += 1
                continue
            if member_name not in requested:
                continue
            if member_name in matched:
                duplicate_member_count += 1
                errors.append(f"Duplicate selected archive member: {member_name}")
                continue
            matched.add(member_name)
            if not member.isfile():
                errors.append(f"Selected member is not a regular file: {member_name}")
                continue
            found.add(member_name)
            try:
                destination = staged_path(image_directory, requested[member_name])
                if destination.exists():
                    if not destination.is_file():
                        raise ValueError(
                            f"Existing destination is not a regular file: {destination}"
                        )
                    existing_count += 1
                elif not dry_run:
                    copy_member(archive, member, destination)
                    extracted_count += 1
            except (OSError, ValueError) as error:
                errors.append(str(error))
            if len(found) % progress_every == 0:
                print(
                    f"Found {len(found):,}/{len(requested):,}; "
                    f"extracted {extracted_count:,}; kept {existing_count:,} existing",
                    flush=True,
                )
            if archive_url is not None and len(found) == len(requested):
                print("All selected images found; closing the remote stream.", flush=True)
                print("Duplicate-member checks cover only the scanned portion of the archive.")
                break

    missing = sorted(set(requested) - found)
    summary = {
        "requested_images": len(requested), "scanned_members": scanned_count,
        "found_regular_members": len(found), "missing_members": len(missing),
        "extracted_images": extracted_count, "kept_existing_images": existing_count,
        "unsafe_members_skipped": unsafe_member_count,
        "duplicate_selected_members": duplicate_member_count,
        "errors": len(errors), "dry_run": dry_run,
    }
    if archive_url is not None and content_length is not None:
        summary["remote_content_length"] = content_length
    print("Extraction summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if missing:
        print("Missing selected members (first 10):", missing[:10])
        print(
            "Check the archive layout and explicit prefix options; "
            "paths are not matched by basename."
        )
    if errors:
        print("Extraction issues (first 10):", errors[:10])
    if existing_count:
        print("Existing files were kept unchanged; validate them before use.")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    archive_options = parser.add_mutually_exclusive_group(required=True)
    archive_options.add_argument(
        "--archive", type=Path,
        help="Exact local tar archive path supplied by the user",
    )
    archive_options.add_argument(
        "--archive-url", help="HTTP or HTTPS URL to stream without saving the full archive",
    )
    parser.add_argument(
        "--list-first", type=int, metavar="N",
        help="Preview the first N member names without extracting; takes precedence over dry-run",
    )
    default_manifest = PROJECT_ROOT / "data/manifests/pilot_image_assets.parquet"
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument(
        "--output", type=Path,
        help="Staging root; defaults to configured pilot_root/images",
    )
    parser.add_argument(
        "--strip-prefix", default="",
        help="Manifest-path prefix to remove for archive matching, e.g. visual_news/",
    )
    parser.add_argument(
        "--member-prefix", default="",
        help="Explicit archive directory prefix to prepend for matching",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan/count selected members without extracting",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    arguments = parser.parse_args(argv)
    if arguments.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if arguments.list_first is not None and arguments.list_first <= 0:
        parser.error("--list-first must be positive")
    if arguments.list_first is not None:
        try:
            preview_archive(arguments.archive, arguments.archive_url, arguments.list_first)
        except (OSError, ValueError, tarfile.TarError, EOFError) as error:
            parser.exit(2, f"Archive preview failed: {error}\n")
        return 0
    image_directory = arguments.output
    if image_directory is None:
        image_directory = default_image_directory()
    try:
        assets = load_asset_manifest(arguments.manifest)
        summary = extract_selected(
            arguments.archive, assets, image_directory,
            arguments.strip_prefix, arguments.member_prefix,
            arguments.dry_run, arguments.progress_every,
            archive_url=arguments.archive_url,
        )
    except (OSError, ValueError, tarfile.TarError, EOFError) as error:
        parser.exit(2, f"Extraction failed: {error}\n")
    return 1 if summary["missing_members"] or summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
