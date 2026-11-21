"""Unit tests on synthetic diffs: python -m unittest discover -s staticeq/tests -t ."""

from __future__ import annotations

import difflib
import os
import tempfile
import textwrap
import unittest

from staticeq import analyze, apply_hunks, parse_patch
from staticeq.datamap import DataMap
from staticeq.segindex import build_index, load_index

MODULE = textwrap.dedent('''\
    """Module docstring."""

    from __future__ import annotations

    import os

    X = 1
    TABLE = {
        "a": lambda args: args[0],
        "b": lambda args: args[1],
    }


    def helper(a, b=2):
        """Helper docstring."""
        total = a + b
        return total


    def outer():
        def inner(x):
            return x * 2

        return inner


    class Base:
        """Base docstring."""

        KEY = "base"
        MAP = {"x": 1}

        def method(self, value):
            return value + 1

        class Nested:
            FLAG = True

            def run(self):
                return "run"


    class Child(Base):
        def method(self, value):
            return super().method(value) * 2
    ''')

FIXTURE = "-- header comment\nSELECT 1;\nSELECT 1;\n\nSELECT a FROM t;\nSELECT a FROM t;\n"
HELPERS = "def load_sql_fixtures(f):\n    return f\n"


def make_diff(path: str, old: str, new: str) -> str:
    lines = difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                 f"a/{path}", f"b/{path}", n=3)
    body = "".join(x if x.endswith("\n") else x + "\n\\ No newline at end of file\n" for x in lines)
    return f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n" + body


class Tree:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "repo")
        self.files = {
            "sqlglot/mod.py": MODULE,
            "sqlglot/__init__.py": "# package\nfrom sqlglot.mod import X\n",
            "tests/helpers.py": HELPERS,
            "tests/test_mod.py": "import unittest\n\n\nclass TestMod(unittest.TestCase):\n"
                                 "    def test_one(self):\n        self.assertTrue(True)\n",
            "tests/fixtures/opt.sql": FIXTURE,
            "tests/fixtures/data.json": '{"a": 1}\n',
            "README.md": "# readme\n",
            "pyproject.toml": "[project]\nname = 'x'\n",
            "benchmarks/bench.py": "print('bench')\n",
        }
        for rel, text in self.files.items():
            full = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(text)
        self.segdir = os.path.join(self.tmp.name, "segments")
        build_index(self.root, self.segdir)
        self.segments = load_index(self.segdir)
        self.map = DataMap({
            "tests": ["tests/test_mod.py::TestMod::test_one", "tests/test_mod.py::TestMod::test_two"],
            "reads": {
                "tests/fixtures/opt.sql": {"tests": [0], "readers": ["tests/helpers.py:load_sql_fixtures"]},
                "tests/fixtures/data.json": {"tests": [1], "readers": ["tests/test_mod.py:TestMod.test_two"]},
            },
            "source_reads": {},
            "listdirs": {"tests/fixtures": {"tests": [1], "readers": ["tests/test_mod.py:x"]}},
            "missing_imports": {"sqlglot.newmod": [0]},
            "imported": ["sqlglot/mod.py", "sqlglot/__init__.py", "tests/helpers.py", "tests/test_mod.py"],
            "all_tests_reads": [],
            "mentions": {},
            "typing_guard": {"ok": True, "sensitive_files": []},
        })

    def run(self, path: str, new: str, *, chunked: bool = True, **kw) -> dict:
        diff = make_diff(path, self.files[path], new)
        return analyze(diff, self.root, data_map=kw.pop("data_map", self.map),
                       segments=self.segments if chunked else None, **kw)

    def both(self, path: str, new: str, **kw) -> dict:
        """Run chunked and whole-file; they must agree on verdict and units."""
        a = self.run(path, new, chunked=True, **kw)
        b = self.run(path, new, chunked=False, **kw)
        ua = sorted((u["qualname"], u["where"]) for u in a["units"])
        ub = sorted((u["qualname"], u["where"]) for u in b["units"])
        assert a["neutral"] == b["neutral"], (a, b)
        assert ua == ub, (ua, ub)
        return a


T: Tree


def setUpModule():
    global T
    T = Tree()


def tearDownModule():
    T.tmp.cleanup()


def edit(text: str, old: str, new: str) -> str:
    assert old in text, old
    return text.replace(old, new, 1)


def units(r: dict) -> list[tuple[str, str]]:
    return sorted((u["qualname"], u["where"]) for u in r["units"])


class TestPatchParsing(unittest.TestCase):
    def test_modify_and_apply(self):
        old = "a\nb\nc\n"
        new = "a\nB\nc\nd\n"
        (fp,) = parse_patch(make_diff("x.txt", old, new))
        self.assertEqual(fp.path, "x.txt")
        self.assertFalse(fp.problems)
        out, mapping = apply_hunks(old.encode(), fp.hunks)
        self.assertEqual(out, new.encode())
        self.assertEqual(mapping[1], 1)
        self.assertEqual(mapping[2], 0)
        self.assertEqual(mapping[3], 3)

    def test_no_newline_at_eof(self):
        old = "a\nb"
        new = "a\nb\nc"
        (fp,) = parse_patch(make_diff("x.txt", old, new))
        out, _ = apply_hunks(old.encode(), fp.hunks)
        self.assertEqual(out, new.encode())

    def test_insert_at_start(self):
        old = "a\nb\n"
        new = "z\na\nb\n"
        (fp,) = parse_patch(make_diff("x.txt", old, new))
        out, _ = apply_hunks(old.encode(), fp.hunks)
        self.assertEqual(out, new.encode())
        removed, inserts = fp.touched_old_lines()
        self.assertEqual(removed, set())
        self.assertEqual(inserts, {0})

    def test_structural_headers(self):
        text = (
            "diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n"
            "diff --git a/n.txt b/n.txt\nnew file mode 100644\nindex 0000000..1111111\n--- /dev/null\n"
            "+++ b/n.txt\n@@ -0,0 +1 @@\n+hi\n"
            "diff --git a/d.txt b/d.txt\ndeleted file mode 100644\nindex 1111111..0000000\n--- a/d.txt\n"
            "+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n"
            "diff --git a/b.bin b/b.bin\nindex 1111111..2222222 100644\nBinary files a/b.bin and b/b.bin differ\n"
            "diff --git a/s.sh b/s.sh\nold mode 100644\nnew mode 100755\n"
            'diff --git "a/sp ace.txt" "b/sp ace.txt"\nindex 1..2 100644\n--- "a/sp ace.txt"\n'
            '+++ "b/sp ace.txt"\n@@ -1 +1 @@\n-x\n+y\n'
        )
        fps = parse_patch(text)
        self.assertEqual(len(fps), 6)
        ren, new, dele, binary, mode, spaced = fps
        self.assertTrue(ren.is_rename)
        self.assertEqual((ren.old_path, ren.new_path), ("old.py", "new.py"))
        self.assertTrue(new.is_new)
        self.assertIsNone(new.old_path)
        self.assertTrue(dele.is_deleted)
        self.assertTrue(binary.is_binary)
        self.assertTrue(mode.mode_change)
        self.assertEqual(spaced.path, "sp ace.txt")
        for fp in fps:
            self.assertFalse(fp.problems, fp)

    def test_bad_hunk_counts(self):
        text = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n a\n-b\n+c\n"
        (fp,) = parse_patch(text)
        self.assertTrue(fp.problems)

    def test_context_mismatch(self):
        (fp,) = parse_patch(make_diff("x.txt", "a\nb\nc\n", "a\nB\nc\n"))
        from staticeq.patch import ApplyError

        with self.assertRaises(ApplyError):
            apply_hunks(b"a\nX\nc\n", fp.hunks)


class TestNeutral(unittest.TestCase):
    P = "sqlglot/mod.py"

    def test_comment_lines(self):
        for old, new in [
            ("    total = a + b\n", "    # explain\n    total = a + b\n"),
            ("    KEY = \"base\"\n", "    # comment\n    KEY = \"base\"\n"),
            ("X = 1\n", "X = 1  # trailing\n"),
            ("class Child(Base):\n", "# a comment above a class\nclass Child(Base):\n"),
            ("        FLAG = True\n", "        FLAG = True\n        # end\n"),
            ('"""Module docstring."""', '# -*- coding: latin-1 -*-\n"""Module docstring."""'),
        ]:
            with self.subTest(new=new):
                r = T.both(self.P, edit(MODULE, old, new))
                self.assertTrue(r["neutral"], r["reasons"])

    def test_blank_lines_and_reformat(self):
        new = edit(MODULE, "def helper(a, b=2):\n", "\n\ndef helper(\n    a,\n    b=2,\n):\n")
        self.assertTrue(T.both(self.P, new)["neutral"])
        new = edit(MODULE, 'MAP = {"x": 1}', 'MAP = {\n        "x": 1,\n    }')
        self.assertTrue(T.both(self.P, new)["neutral"])

    def test_docstring(self):
        new = edit(MODULE, '"""Helper docstring."""', '"""Helper docstring, reworded."""')
        r = T.both(self.P, new)
        self.assertTrue(r["neutral"])
        self.assertEqual(r["files"][0]["level"], "docstring-only")
        strict = T.run(self.P, new, docstrings_neutral=False)
        self.assertFalse(strict["neutral"])
        self.assertEqual(units(strict), [("helper", "docstring")])

    def test_docstring_with_new_warning_is_not_neutral(self):
        new = edit(MODULE, '"""Helper docstring."""', '"""Helper \\d docstring."""')
        r = T.run(self.P, new)
        self.assertFalse(r["neutral"])

    def test_no_map_still_proves_code(self):
        new = edit(MODULE, "    total = a + b\n", "    # explain\n    total = a + b\n")
        self.assertTrue(T.run(self.P, new, data_map=None)["neutral"])


class TestUnits(unittest.TestCase):
    P = "sqlglot/mod.py"

    def check(self, old: str, new: str, expected: list[tuple[str, str]]):
        r = T.both(self.P, edit(MODULE, old, new))
        self.assertFalse(r["neutral"])
        self.assertEqual(units(r), expected)
        return r

    def test_function_body(self):
        r = self.check("total = a + b", "total = a - b", [("helper", "body")])
        u = r["units"][0]
        self.assertEqual(u["module"], "sqlglot.mod")
        self.assertEqual(u["def_line"], 14)
        self.assertEqual(u["old_lines"], [14, 17])

    def test_string_constant(self):
        self.check('return "run"', 'return "walk"', [("Base.Nested.run", "body")])

    def test_default_and_decorator(self):
        self.check("def helper(a, b=2):", "def helper(a, b=3):", [("helper", "header")])
        self.check("    def method(self, value):\n        return value + 1",
                   "    @staticmethod\n    def method(self, value):\n        return value + 1",
                   [("Base.method", "header")])

    def test_class_attribute(self):
        r = self.check('KEY = "base"', 'KEY = "other"', [("Base", "class-body")])
        self.assertEqual(r["units"][0]["names"], ["KEY"])
        self.check("FLAG = True", "FLAG = False", [("Base.Nested", "class-body")])

    def test_method_added(self):
        r = self.check("class Child(Base):\n", "class Child(Base):\n    def extra(self):\n        return 1\n\n",
                       [("Child", "class-body")])
        self.assertIn("extra", r["units"][0]["names"])

    def test_class_header(self):
        self.check("class Child(Base):", "class Child(Base, object):", [("Child", "class-header")])

    def test_module_level(self):
        r = self.check("X = 1", "X = 2", [("", "module-level")])
        self.assertEqual(r["units"][0]["names"], ["X"])
        self.check("import os\n", "import os\nimport sys\n", [("", "module-level")])

    def test_lambda_in_table(self):
        r = self.check('"b": lambda args: args[1]', '"b": lambda args: args[2]', [("<lambda>", "body")])
        self.assertEqual(r["units"][0]["old_lines"], [10, 10])

    def test_nested_function(self):
        self.check("return x * 2", "return x * 3", [("outer.<locals>.inner", "body")])

    def test_super_method(self):
        self.check("super().method(value) * 2", "super().method(value) * 3", [("Child.method", "body")])

    def test_syntax_error_from_comment(self):
        new = edit(MODULE, "    total = a + b\n", "    total = a + \\\n    # oops\n    b\n")
        r = T.run(self.P, new)
        self.assertFalse(r["neutral"])
        self.assertTrue(any("import fails" in u["detail"] for u in r["units"]))

    def test_encoding_cookie(self):
        # the cookie re-decodes the UTF-8 bytes of a code string: behavior changes
        new = edit(MODULE, '"""Module docstring."""', '# -*- coding: latin-1 -*-\n"""Module docstring."""')
        new = edit(new, 'KEY = "base"', 'KEY = "b\\u00e4se"'.encode().decode("unicode_escape"))
        self.assertFalse(T.run(self.P, new)["neutral"])
        # only a docstring is re-decoded: docstring-only
        new = edit(MODULE, '"""Module docstring."""',
                   '# -*- coding: latin-1 -*-\n"""Module d\\u00f6cstring."""'.encode().decode("unicode_escape"))
        r = T.run(self.P, new)
        self.assertEqual(r["files"][0]["level"], "docstring-only")

    def test_test_file_unit_has_test_id(self):
        path = "tests/test_mod.py"
        new = T.files[path].replace("assertTrue(True)", "assertTrue(False)")
        r = T.run(path, new)
        self.assertFalse(r["neutral"])
        self.assertEqual(r["units"][0]["test_ids"], ["tests/test_mod.py::TestMod::test_one"])


class TestDataFiles(unittest.TestCase):
    def test_sql_comment_lines_are_neutral(self):
        new = FIXTURE.replace("SELECT 1;\nSELECT 1;\n", "SELECT 1;\n-- why\nSELECT 1;\n\n")
        r = T.run("tests/fixtures/opt.sql", new)
        self.assertTrue(r["neutral"], r["reasons"])

    def test_sql_statement_change(self):
        r = T.run("tests/fixtures/opt.sql", FIXTURE.replace("SELECT a FROM t;\nSELECT a", "SELECT b FROM t;\nSELECT a"))
        self.assertFalse(r["neutral"])
        self.assertEqual(r["data_files"][0]["tests"], ["tests/test_mod.py::TestMod::test_one"])

    def test_inline_sql_comment_is_not_neutral(self):
        r = T.run("tests/fixtures/opt.sql", FIXTURE.replace("SELECT 1;\nSELECT 1;", "SELECT 1; -- x\nSELECT 1;"))
        self.assertFalse(r["neutral"])

    def test_other_reader(self):
        r = T.run("tests/fixtures/data.json", '{"a": 2}\n')
        self.assertFalse(r["neutral"])
        self.assertEqual(r["data_files"][0]["tests"], ["tests/test_mod.py::TestMod::test_two"])

    def test_unread_files(self):
        self.assertTrue(T.run("README.md", "# readme!\n")["neutral"])
        self.assertTrue(T.run("benchmarks/bench.py", "print('bench2')\n")["neutral"])

    def test_no_map_is_conservative(self):
        self.assertFalse(T.run("README.md", "# readme!\n", data_map=None)["neutral"])

    def test_config(self):
        self.assertFalse(T.run("pyproject.toml", "[project]\nname = 'x'\n[tool.pytest.ini_options]\naddopts = '-x'\n")["neutral"])
        self.assertFalse(T.run("pyproject.toml", "[project\nname = 'x'\n")["neutral"])
        self.assertTrue(T.run("pyproject.toml", "[project]\nname = 'y'\n")["neutral"])

    def test_structural(self):
        new_file = ("diff --git a/tests/fixtures/new.sql b/tests/fixtures/new.sql\nnew file mode 100644\n"
                    "index 0000000..1111111\n--- /dev/null\n+++ b/tests/fixtures/new.sql\n@@ -0,0 +1 @@\n+SELECT 1;\n")
        r = analyze(new_file, T.root, data_map=T.map, segments=T.segments)
        self.assertFalse(r["neutral"])
        self.assertEqual(r["data_files"][0]["dir_listers"], ["tests/test_mod.py::TestMod::test_two"])
        new_mod = new_file.replace("tests/fixtures/new.sql", "sqlglot/newmod.py")
        r = analyze(new_mod, T.root, data_map=T.map, segments=T.segments)
        self.assertFalse(r["neutral"])
        self.assertEqual(r["data_files"][0]["tests"], ["tests/test_mod.py::TestMod::test_one"])
        self.assertEqual(r["units"][0]["where"], "file-added")
        conftest = new_file.replace("tests/fixtures/new.sql", "tests/conftest.py")
        self.assertFalse(analyze(conftest, T.root, data_map=T.map)["neutral"])

    def test_garbage(self):
        self.assertFalse(analyze("", T.root)["neutral"])
        self.assertFalse(analyze("not a diff at all\n", T.root)["neutral"])
        self.assertFalse(analyze(b"\xff\xfe binary", T.root)["neutral"])

    def test_multi_file(self):
        a = make_diff("sqlglot/mod.py", MODULE, edit(MODULE, "    total = a + b\n", "    # c\n    total = a + b\n"))
        b = make_diff("tests/fixtures/opt.sql", FIXTURE, FIXTURE + "-- trailing\n")
        r = analyze(a + b, T.root, data_map=T.map, segments=T.segments)
        self.assertTrue(r["neutral"], r["reasons"])
        c = make_diff("sqlglot/mod.py", MODULE, edit(MODULE, "X = 1", "X = 3"))
        r = analyze(b + c, T.root, data_map=T.map, segments=T.segments)
        self.assertFalse(r["neutral"])
        self.assertEqual(units(r), [("", "module-level")])


if __name__ == "__main__":
    unittest.main()
