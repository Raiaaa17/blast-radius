"""Download sqlglot at the pinned commit from its own GitHub repository.

Usage: python fetch_sqlglot.py <sha> <dest>
"""

import io
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


def main() -> None:
    sha, dest = sys.argv[1], Path(sys.argv[2])
    url = f"https://github.com/tobymao/sqlglot/archive/{sha}.tar.gz"
    data = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            break
        except Exception as error:  # network hiccup: retry
            print(f"download failed ({error}); retrying", file=sys.stderr)
            time.sleep(2 + 3 * attempt)
    if data is None:
        raise SystemExit("could not download sqlglot")
    tmp = dest.parent / (dest.name + ".extract")
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extractall(tmp, filter="data")
    (root,) = list(tmp.iterdir())
    shutil.rmtree(dest, ignore_errors=True)
    root.rename(dest)
    tmp.rmdir()
    print(f"sqlglot {sha} -> {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
