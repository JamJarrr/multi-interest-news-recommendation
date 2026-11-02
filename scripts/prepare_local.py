"""Normalize archive layout and preserve the original collaborator results."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    raw = ROOT / "data/raw"
    reference = ROOT / "results/reference"
    reference.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / "results").glob("*.json"):
        destination = reference / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
    manifest = {}
    for split in ("train", "dev"):
        path = raw / "zips" / f"MINDsmall_{split}.zip"
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Bad archive member: {bad}")
            records = {}
            for filename in ("news.tsv", "behaviors.tsv"):
                matches = [n for n in archive.namelist() if Path(n).name == filename]
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one {filename} in {path}")
                content = archive.read(matches[0])
                destination = raw / f"MINDsmall_{split}" / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and destination.read_bytes() != content:
                    raise ValueError(f"Refusing to overwrite different data: {destination}")
                destination.write_bytes(content)
                records[filename] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            manifest[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "bytes": path.stat().st_size, "members": records}
    (ROOT / "results/local_data").mkdir(exist_ok=True)
    (ROOT / "results/local_data/source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()

