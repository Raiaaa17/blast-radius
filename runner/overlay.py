"""A cheap writable view of the read-only pristine sqlglot checkout.

The lane root (e.g. /tmp/br-runner/lane0/root) starts as one symlink per top-level entry
of the pristine checkout. Applying a patch "materializes" only the directories on the
path to each changed file (a real directory whose entries are symlinks back to the
pristine ones) and writes the changed files as real files, so a patch costs a few
hundred symlinks at most and the 64 MB /tmp is never filled with a full copy.

Every path that tests read is resolved through the lane root, so changed fixtures and
data files are seen by the tests without redirecting any read. Python imports
`__file__`-relative paths unresolved, so modules, fixtures and inspect/linecache see the
patched tree. Test files that call Path.resolve()/realpath on their own path (which
would escape to the pristine tree) are kept as real copies (see `copy_files`).

The stale bytecode of a changed module is hidden by materializing its `__pycache__`
without that module's .pyc files.
"""

from __future__ import annotations

import os
import shutil
import stat

from .patch import NewFile, PatchError

SKIP_TOP = {".git"}


def _check_rel(path: str) -> list[str]:
    parts = [p for p in path.split("/") if p not in ("", ".")]
    if not parts or path.startswith("/") or any(p == ".." for p in parts):
        raise PatchError(f"unsafe path in patch: {path!r}")
    if parts[0] in SKIP_TOP:
        raise PatchError(f"patch touches {parts[0]}: {path!r}")
    return parts


class Overlay:
    def __init__(self, pristine: str, root: str, copy_files: tuple[str, ...] = ()):
        self.pristine = os.path.abspath(pristine)
        self.root = root
        self.copy_files = tuple(copy_files)
        self.dirty = True
        self.top = sorted(n for n in os.listdir(self.pristine) if n not in SKIP_TOP)

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Make the lane root a pristine view again (cheap: ~20 symlinks)."""
        if os.path.lexists(self.root):
            shutil.rmtree(self.root, ignore_errors=True)
            if os.path.lexists(self.root):  # something unremovable; start elsewhere
                raise OSError(f"cannot reset overlay {self.root}")
        os.makedirs(self.root)
        for name in self.top:
            os.symlink(os.path.join(self.pristine, name), os.path.join(self.root, name))
        for rel in self.copy_files:
            src = os.path.join(self.pristine, rel)
            if os.path.isfile(src):
                d = self._materialize_dir(rel.split("/")[:-1])
                dst = os.path.join(d, rel.split("/")[-1])
                os.unlink(dst)
                shutil.copy2(src, dst)
        self.dirty = False

    # ------------------------------------------------------------------ helpers
    def _materialize_dir(self, parts: list[str]) -> str:
        cur = self.root
        for comp in parts:
            p = os.path.join(cur, comp)
            if os.path.islink(p):
                target = os.readlink(p)
                if os.path.isdir(target):
                    os.unlink(p)
                    os.mkdir(p)
                    for child in os.listdir(target):
                        os.symlink(os.path.join(target, child), os.path.join(p, child))
                else:
                    raise PatchError(f"path component is a file: {comp!r}")
            elif not os.path.exists(p):
                os.mkdir(p)
            elif not os.path.isdir(p):
                raise PatchError(f"path component is a file: {comp!r}")
            cur = p
        return cur

    def _hide_pyc(self, dir_abs: str, filename: str) -> None:
        if not filename.endswith(".py"):
            return
        pc = os.path.join(dir_abs, "__pycache__")
        if not os.path.lexists(pc):
            return
        if os.path.islink(pc):
            target = os.readlink(pc)
            os.unlink(pc)
            os.mkdir(pc)
            if os.path.isdir(target):
                for child in os.listdir(target):
                    os.symlink(os.path.join(target, child), os.path.join(pc, child))
        stem = filename[:-3] + "."
        for child in os.listdir(pc):
            if child.startswith(stem):
                os.unlink(os.path.join(pc, child))

    # ------------------------------------------------------------------ apply
    def apply(self, files: dict[str, NewFile | None]) -> None:
        """Write the patch result into the lane root. Deletions first, then writes."""
        self.dirty = True
        for path, nf in files.items():
            if nf is not None:
                continue
            parts = _check_rel(path)
            d = self._materialize_dir(parts[:-1])
            leaf = os.path.join(d, parts[-1])
            if os.path.lexists(leaf):
                if os.path.isdir(leaf) and not os.path.islink(leaf):
                    shutil.rmtree(leaf)
                else:
                    os.unlink(leaf)
            self._hide_pyc(d, parts[-1])
        for path, nf in files.items():
            if nf is None:
                continue
            parts = _check_rel(path)
            d = self._materialize_dir(parts[:-1])
            leaf = os.path.join(d, parts[-1])
            if os.path.lexists(leaf):
                if os.path.isdir(leaf) and not os.path.islink(leaf):
                    raise PatchError(f"patch writes a file over a directory: {path!r}")
                os.unlink(leaf)
            if nf.mode == "120000":
                os.symlink(nf.content.decode("utf-8", "surrogateescape"), leaf)
            else:
                with open(leaf, "wb") as f:
                    f.write(nf.content)
                mode = 0o755 if (nf.mode or "").endswith("755") else 0o644
                os.chmod(leaf, mode)
            self._hide_pyc(d, parts[-1])

    def read_pristine(self, path: str) -> bytes | None:
        """Pristine content of a repo-relative path (None if it does not exist)."""
        try:
            parts = _check_rel(path)
        except PatchError:
            return None
        p = os.path.join(self.pristine, *parts)
        try:
            st = os.lstat(p)
        except OSError:
            return None
        if stat.S_ISLNK(st.st_mode):
            return os.readlink(p).encode("utf-8", "surrogateescape")
        if not stat.S_ISREG(st.st_mode):
            return None
        with open(p, "rb") as f:
            return f.read()


def resolve_sensitive_tests(pristine: str) -> tuple[str, ...]:
    """Test files that resolve their own path (they must be real copies in the overlay)."""
    out = []
    tests = os.path.join(pristine, "tests")
    for dirpath, dirnames, filenames in os.walk(tests):
        dirnames[:] = [d for d in dirnames if d not in ("fixtures", "__pycache__")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "rb") as f:
                    src = f.read()
            except OSError:
                continue
            if b".resolve(" in src or b"realpath" in src or b"readlink" in src:
                out.append(os.path.relpath(p, pristine).replace(os.sep, "/"))
    return tuple(sorted(out))
