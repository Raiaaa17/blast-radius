"""staticeq: static analysis of a sqlglot patch.

    from staticeq import analyze
    verdict = analyze(patch_text, "/path/to/pristine/sqlglot")

See README.md for the result shape and the evidence behind each rule.
"""

from .analyze import analyze, module_name
from .datamap import DataMap, load_map
from .patch import FilePatch, Hunk, apply_hunks, parse_patch

__all__ = ["analyze", "module_name", "DataMap", "load_map", "FilePatch", "Hunk", "apply_hunks",
           "parse_patch"]
