"""Download WM-811K while enforcing any previously recorded source checksums."""
from pathlib import Path
import hashlib
import json
import os
import urllib.request
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
URL = "https://www.kaggle.com/api/v1/datasets/download/qingyi/wm811k-wafer-map"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / "wm811k.zip"
    provenance_path = RAW / "provenance.json"
    # A Git clone contains provenance.json but deliberately omits the dataset.
    # Its checksums must constrain a new download, not be replaced by that download.
    previous = json.loads(provenance_path.read_text(encoding="utf-8-sig")) if provenance_path.exists() else None
    if previous is not None:
        for key in ["archive_sha256", "pickle_sha256"]:
            value = previous.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise RuntimeError(f"Existing provenance lacks a valid {key}; refusing to replace it.")
    if archive.exists() and previous is not None:
        if previous["archive_sha256"] != sha256(archive):
            raise RuntimeError("Existing archive does not match its recorded checksum.")
        print("Using checksum-verified existing archive", flush=True)
    else:
        request = urllib.request.Request(URL, headers={"User-Agent": "WaferMapDefense/1.0"})
        part = archive.with_suffix(".zip.part")
        print("Downloading WM-811K from its Kaggle dataset endpoint", flush=True)
        try:
            with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as target:
                total = 0
                last_report = 0
                while chunk := response.read(4 * 1024 * 1024):
                    target.write(chunk)
                    total += len(chunk)
                    if total - last_report >= 32 * 1024 * 1024:
                        print(f"Downloaded {total / 1024**2:.0f} MiB", flush=True)
                        last_report = total
            if not zipfile.is_zipfile(part):
                raise RuntimeError("The dataset endpoint did not return a ZIP archive.")
            downloaded_hash = sha256(part)
            if previous is not None and downloaded_hash != previous["archive_sha256"]:
                raise RuntimeError("Downloaded archive does not match the recorded checksum; provenance is unchanged.")
            os.replace(part, archive)
        finally:
            part.unlink(missing_ok=True)
        # Redirect URLs may contain expiring signed tokens; do not persist them.
    extracted = RAW / "LSWMD.pkl"
    extracted_part = extracted.with_suffix(".pkl.part")
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [m for m in bundle.infolist() if Path(m.filename).name == "LSWMD.pkl"]
            if len(members) != 1:
                raise RuntimeError("Expected exactly one LSWMD.pkl in the archive.")
            member = members[0]
            print(f"Extracting LSWMD.pkl ({member.file_size / 1024**3:.2f} GiB); ZIP CRC checked while reading", flush=True)
            with bundle.open(member) as source, extracted_part.open("wb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    target.write(chunk)
        pickle_hash = sha256(extracted_part)
        if previous is not None and pickle_hash != previous["pickle_sha256"]:
            raise RuntimeError("Extracted pickle does not match its recorded checksum; existing pickle and provenance are unchanged.")
        os.replace(extracted_part, extracted)
    finally:
        extracted_part.unlink(missing_ok=True)
    if previous is None:
        previous = {"source_page": "https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map",
                    "download_url": URL,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
                    "pickle_sha256": pickle_hash, "pickle_bytes": extracted.stat().st_size,
                    "zip_member": member.filename, "zip_crc_verified": True}
        provenance_path.write_text(json.dumps(previous, indent=2), encoding="utf-8")
    # Leave an existing provenance file byte-for-byte intact, including its original timestamp.
    print(json.dumps(previous, indent=2), flush=True)


if __name__ == "__main__":
    main()
