"""Dump the effective (inherited) dispatch tables of pristine sqlglot as JSON.

Run inside the reference image with the pristine tree on PYTHONPATH (gen_mutations.py
does this through docker). The dispatch-shape mutations need what a class *inherits*,
which the source text of one module does not show:

- generators: every Generator subclass defined in sqlglot/generators/*.py, with the
  expression classes in its effective TRANSFORMS and the `*_sql` names in dir(cls);
- parsers: every Parser subclass in sqlglot/parsers/*.py with its effective
  FUNCTIONS / FUNCTION_PARSERS / NO_PAREN_FUNCTION_PARSERS keys;
- expressions: every Func subclass with its dispatch key, SQL names and arg names.
"""

import importlib
import inspect
import json
import pkgutil
import sys

import sqlglot
from sqlglot import exp, generator, parser


def defined_here(module):
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__
    ]


def main():
    out = {"generators": [], "parsers": [], "funcs": []}
    for package, base in (("sqlglot.generators", generator.Generator), ("sqlglot.parsers", parser.Parser)):
        pkg = importlib.import_module(package)
        for info in pkgutil.iter_modules(pkg.__path__):
            module = importlib.import_module(f"{package}.{info.name}")
            for cls in defined_here(module):
                if not issubclass(cls, base) or cls is base:
                    continue
                entry = {
                    "module": module.__name__,
                    "file": module.__file__.split("/src/", 1)[-1],
                    "class": cls.__name__,
                }
                if base is generator.Generator:
                    entry["transforms"] = sorted(k.__name__ for k in cls.TRANSFORMS if isinstance(k, type))
                    entry["sql_methods"] = sorted(n for n in dir(cls) if n.endswith("_sql") and not n.startswith("_"))
                    entry["own_transforms"] = "TRANSFORMS" in vars(cls)
                else:
                    entry["functions"] = sorted(k for k in cls.FUNCTIONS if isinstance(k, str))
                    entry["function_parsers"] = sorted(k for k in cls.FUNCTION_PARSERS if isinstance(k, str))
                    entry["no_paren"] = sorted(
                        k for k in getattr(cls, "NO_PAREN_FUNCTION_PARSERS", {}) if isinstance(k, str)
                    )
                    entry["own_functions"] = "FUNCTIONS" in vars(cls)
                out["generators" if base is generator.Generator else "parsers"].append(entry)
    for key, cls in sorted(exp.EXPR_CLASSES.items()):
        if isinstance(cls, type) and issubclass(cls, exp.Func):
            try:
                names = cls.sql_names()
            except Exception:
                names = []
            out["funcs"].append(
                {
                    "key": key,
                    "class": cls.__name__,
                    "sql_names": names,
                    "args": list(cls.arg_types),
                    "var_len": bool(getattr(cls, "is_var_len_args", False)),
                }
            )
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
