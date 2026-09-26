"""Benchmark tasks for `k2loop bench`: small, realistic Python repo edits.

Each task: starting files, the instruction the model sees, one visible test (so `.k2loop/check.sh` has something to
run, like a real project), HIDDEN tests the model never sees (they decide pass/fail), and a reference solution
(tests/test_bench_tasks.py checks that the solution passes and the starting code fails, so every task measures
something). Standard library only: no dependency installs during the benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent


@dataclass
class BenchTask:
    name: str
    level: str                      # easy | medium | hard
    prompt: str
    files: dict[str, str]
    hidden: dict[str, str]
    solution: dict[str, str]
    hint_files: list[str] = field(default_factory=list)


def _d(s: str) -> str:
    return dedent(s).lstrip("\n")


TASKS: list[BenchTask] = [
    BenchTask(
        name="csv_stats", level="easy",
        prompt=_d("""
            In stats.py add a function `column_stats(path, column)` that reads the CSV file at `path` (first row is the
            header) and returns a dict {"min": float, "max": float, "mean": float, "count": int} for the numeric values
            of `column`. Empty cells are skipped. Raise KeyError if the column does not exist and ValueError if the
            column has no values. Use the existing `read_rows` helper. Add a test for it in tests/test_stats.py.
        """),
        files={
            "stats.py": _d("""
                import csv


                def read_rows(path):
                    with open(path, newline="", encoding="utf-8") as fh:
                        return list(csv.DictReader(fh))
            """),
            "tests/test_stats.py": _d("""
                from stats import read_rows


                def test_read_rows(tmp_path):
                    p = tmp_path / "a.csv"
                    p.write_text("x,y\\n1,2\\n")
                    assert read_rows(p) == [{"x": "1", "y": "2"}]
            """),
        },
        hidden={"tests/test_hidden_csv_stats.py": _d("""
            import pytest
            from stats import column_stats


            def _csv(tmp_path, text):
                p = tmp_path / "d.csv"
                p.write_text(text)
                return p


            def test_basic(tmp_path):
                s = column_stats(_csv(tmp_path, "a,b\\n1,x\\n3,y\\n5,z\\n"), "a")
                assert s == {"min": 1.0, "max": 5.0, "mean": 3.0, "count": 3}


            def test_skips_empty_and_floats(tmp_path):
                s = column_stats(_csv(tmp_path, "a,b\\n1.5,x\\n,y\\n2.5,z\\n"), "a")
                assert s["count"] == 2 and s["mean"] == pytest.approx(2.0) and s["min"] == 1.5


            def test_missing_column(tmp_path):
                with pytest.raises(KeyError):
                    column_stats(_csv(tmp_path, "a\\n1\\n"), "nope")


            def test_no_values(tmp_path):
                with pytest.raises(ValueError):
                    column_stats(_csv(tmp_path, "a,b\\n,1\\n"), "a")
        """)},
        solution={"stats.py": _d("""
            import csv


            def read_rows(path):
                with open(path, newline="", encoding="utf-8") as fh:
                    return list(csv.DictReader(fh))


            def column_stats(path, column):
                rows = read_rows(path)
                if not rows or column not in rows[0]:
                    raise KeyError(column)
                vals = [float(r[column]) for r in rows if (r.get(column) or "").strip()]
                if not vals:
                    raise ValueError(f"no values in {column}")
                return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals), "count": len(vals)}
        """)},
        hint_files=["stats.py", "tests/test_stats.py"],
    ),
    BenchTask(
        name="cli_flags", level="easy",
        prompt=_d("""
            wc.py is a small word-count CLI. `main(argv)` currently prints the number of lines of the file. Add three
            flags: `--lines` (number of lines), `--words` (number of whitespace-separated words) and `--chars`
            (number of characters). Print only the requested counts on one line, always in the order lines, words,
            chars, separated by one space. With no flag, keep printing only the line count. `main` must return 0.
        """),
        files={
            "wc.py": _d("""
                import argparse


                def main(argv=None):
                    ap = argparse.ArgumentParser()
                    ap.add_argument("path")
                    args = ap.parse_args(argv)
                    with open(args.path, encoding="utf-8") as fh:
                        text = fh.read()
                    print(len(text.splitlines()))
                    return 0


                if __name__ == "__main__":
                    raise SystemExit(main())
            """),
            "tests/test_wc.py": _d("""
                from wc import main


                def test_lines(tmp_path, capsys):
                    p = tmp_path / "t.txt"
                    p.write_text("a b\\nc\\n")
                    assert main([str(p)]) == 0
                    assert capsys.readouterr().out.strip() == "2"
            """),
        },
        hidden={"tests/test_hidden_cli_flags.py": _d("""
            from wc import main


            def _run(tmp_path, capsys, *flags):
                p = tmp_path / "t.txt"
                p.write_text("hello big world\\nsecond line\\n")
                assert main([*flags, str(p)]) == 0
                return capsys.readouterr().out.split()


            def test_default(tmp_path, capsys):
                assert _run(tmp_path, capsys) == ["2"]


            def test_words(tmp_path, capsys):
                assert _run(tmp_path, capsys, "--words") == ["5"]


            def test_chars(tmp_path, capsys):
                assert _run(tmp_path, capsys, "--chars") == ["28"]


            def test_combined_order(tmp_path, capsys):
                assert _run(tmp_path, capsys, "--chars", "--lines", "--words") == ["2", "5", "28"]
        """)},
        solution={"wc.py": _d("""
            import argparse


            def main(argv=None):
                ap = argparse.ArgumentParser()
                ap.add_argument("path")
                ap.add_argument("--lines", action="store_true")
                ap.add_argument("--words", action="store_true")
                ap.add_argument("--chars", action="store_true")
                args = ap.parse_args(argv)
                with open(args.path, encoding="utf-8") as fh:
                    text = fh.read()
                want_lines = args.lines or not (args.words or args.chars)
                out = []
                if want_lines:
                    out.append(len(text.splitlines()))
                if args.words:
                    out.append(len(text.split()))
                if args.chars:
                    out.append(len(text))
                print(" ".join(map(str, out)))
                return 0


            if __name__ == "__main__":
                raise SystemExit(main())
        """)},
        hint_files=["wc.py"],
    ),
    BenchTask(
        name="date_bugfix", level="medium",
        prompt=_d("""
            Users report that `add_months` in dates.py gives wrong results or crashes. Fix it. Expected behaviour:
            `add_months(date, n)` moves the date by n months (n may be negative); if the day does not exist in the
            target month, use the last day of that month (Jan 31 + 1 month = Feb 28, or Feb 29 in leap years).
            Do not use third-party libraries. Add a regression test.
        """),
        files={
            "dates.py": _d("""
                import datetime


                def add_months(d, n):
                    month = d.month + n
                    year = d.year + month // 12
                    month = month % 12
                    return datetime.date(year, month, d.day)
            """),
            "tests/test_dates.py": _d("""
                import datetime

                from dates import add_months


                def test_simple():
                    assert add_months(datetime.date(2024, 3, 15), 1) == datetime.date(2024, 4, 15)
            """),
        },
        hidden={"tests/test_hidden_date_bugfix.py": _d("""
            import datetime as dt

            from dates import add_months


            def test_year_wrap():
                assert add_months(dt.date(2023, 11, 10), 2) == dt.date(2024, 1, 10)
                assert add_months(dt.date(2023, 12, 10), 1) == dt.date(2024, 1, 10)


            def test_negative():
                assert add_months(dt.date(2024, 1, 10), -1) == dt.date(2023, 12, 10)
                assert add_months(dt.date(2024, 3, 31), -13) == dt.date(2023, 2, 28)


            def test_clamp_and_leap():
                assert add_months(dt.date(2023, 1, 31), 1) == dt.date(2023, 2, 28)
                assert add_months(dt.date(2024, 1, 31), 1) == dt.date(2024, 2, 29)
                assert add_months(dt.date(2024, 5, 31), 1) == dt.date(2024, 6, 30)


            def test_zero_and_big():
                assert add_months(dt.date(2024, 2, 29), 0) == dt.date(2024, 2, 29)
                assert add_months(dt.date(2024, 2, 29), 48) == dt.date(2028, 2, 29)
        """)},
        solution={"dates.py": _d("""
            import calendar
            import datetime


            def add_months(d, n):
                idx = d.year * 12 + (d.month - 1) + n
                year, month = divmod(idx, 12)
                month += 1
                day = min(d.day, calendar.monthrange(year, month)[1])
                return datetime.date(year, month, day)
        """)},
        hint_files=["dates.py"],
    ),
    BenchTask(
        name="lru_cache", level="medium",
        prompt=_d("""
            Implement class `LRUCache` in lru.py: `LRUCache(capacity)` (capacity >= 1, else ValueError),
            `get(key, default=None)` returns the value and marks the key as most recently used, `put(key, value)`
            inserts or updates (an update also marks it most recently used) and, when the cache is over capacity,
            evicts the least recently used key. `len(cache)` returns the number of items and `key in cache` works
            WITHOUT changing the order. Both get and put must be O(1). Add tests.
        """),
        files={
            "lru.py": _d("""
                class LRUCache:
                    \"\"\"Least-recently-used cache. TODO: implement.\"\"\"
            """),
            "tests/test_lru.py": _d("""
                from lru import LRUCache


                def test_exists():
                    assert LRUCache
            """),
        },
        hidden={"tests/test_hidden_lru_cache.py": _d("""
            import pytest
            from lru import LRUCache


            def test_eviction_order():
                c = LRUCache(2)
                c.put("a", 1)
                c.put("b", 2)
                assert c.get("a") == 1
                c.put("c", 3)
                assert "b" not in c and "a" in c and "c" in c and len(c) == 2


            def test_update_refreshes():
                c = LRUCache(2)
                c.put("a", 1)
                c.put("b", 2)
                c.put("a", 10)
                c.put("c", 3)
                assert c.get("a") == 10 and c.get("b") is None and c.get("b", "x") == "x"


            def test_contains_does_not_refresh():
                c = LRUCache(2)
                c.put("a", 1)
                c.put("b", 2)
                assert "a" in c
                c.put("c", 3)
                assert "a" not in c


            def test_capacity_validation():
                with pytest.raises(ValueError):
                    LRUCache(0)
        """)},
        solution={"lru.py": _d("""
            from collections import OrderedDict


            class LRUCache:
                def __init__(self, capacity):
                    if capacity < 1:
                        raise ValueError("capacity must be >= 1")
                    self.capacity = capacity
                    self._d = OrderedDict()

                def get(self, key, default=None):
                    if key not in self._d:
                        return default
                    self._d.move_to_end(key)
                    return self._d[key]

                def put(self, key, value):
                    self._d[key] = value
                    self._d.move_to_end(key)
                    while len(self._d) > self.capacity:
                        self._d.popitem(last=False)

                def __len__(self):
                    return len(self._d)

                def __contains__(self, key):
                    return key in self._d
        """)},
        hint_files=["lru.py"],
    ),
    BenchTask(
        name="deep_merge", level="medium",
        prompt=_d("""
            In config_merge.py implement `deep_merge(base, override)` that returns a NEW dict (inputs must not be
            modified): keys of `override` win; when both values are dicts they are merged recursively; lists and all
            other values are replaced, not merged; an override value of None deletes the key from the result.
            Add tests.
        """),
        files={
            "config_merge.py": _d("""
                def deep_merge(base, override):
                    raise NotImplementedError
            """),
            "tests/test_config_merge.py": _d("""
                from config_merge import deep_merge


                def test_callable():
                    assert callable(deep_merge)
            """),
        },
        hidden={"tests/test_hidden_deep_merge.py": _d("""
            import copy

            from config_merge import deep_merge


            def test_recursive_and_replace():
                base = {"a": 1, "b": {"c": 2, "d": [1, 2]}, "e": {"f": 1}}
                over = {"b": {"d": [3], "x": 9}, "e": 5}
                assert deep_merge(base, over) == {"a": 1, "b": {"c": 2, "d": [3], "x": 9}, "e": 5}


            def test_none_deletes():
                assert deep_merge({"a": 1, "b": {"c": 2, "d": 3}}, {"a": None, "b": {"c": None}}) == {"b": {"d": 3}}


            def test_inputs_untouched():
                base = {"a": {"b": {"c": 1}}}
                over = {"a": {"b": {"d": 2}}}
                b0, o0 = copy.deepcopy(base), copy.deepcopy(over)
                out = deep_merge(base, over)
                out["a"]["b"]["z"] = 1
                assert base == b0 and over == o0


            def test_new_nested_dict_is_copied():
                over = {"n": {"m": 1}}
                out = deep_merge({}, over)
                out["n"]["m"] = 2
                assert over == {"n": {"m": 1}}
        """)},
        solution={"config_merge.py": _d("""
            import copy


            def deep_merge(base, override):
                out = copy.deepcopy(base)
                for k, v in override.items():
                    if v is None:
                        out.pop(k, None)
                    elif isinstance(v, dict) and isinstance(out.get(k), dict):
                        out[k] = deep_merge(out[k], v)
                    else:
                        out[k] = copy.deepcopy(v)
                return out
        """)},
        hint_files=["config_merge.py"],
    ),
    BenchTask(
        name="rename_across_modules", level="medium",
        prompt=_d("""
            Rename the function `calc_total` in shop/pricing.py to `order_total`, update every caller in the package,
            and keep `calc_total` working as a deprecated alias that emits a DeprecationWarning when called.
            Also add a `discount` keyword argument to `order_total` (a fraction between 0 and 1, default 0, applied to
            the total; raise ValueError outside that range).
        """),
        files={
            "shop/__init__.py": "",
            "shop/pricing.py": _d("""
                def calc_total(items):
                    return round(sum(price * qty for price, qty in items), 2)
            """),
            "shop/cart.py": _d("""
                from shop.pricing import calc_total


                class Cart:
                    def __init__(self):
                        self.items = []

                    def add(self, price, qty=1):
                        self.items.append((price, qty))

                    def total(self):
                        return calc_total(self.items)
            """),
            "tests/test_cart.py": _d("""
                from shop.cart import Cart


                def test_total():
                    c = Cart()
                    c.add(2.5, 2)
                    assert c.total() == 5.0
            """),
        },
        hidden={"tests/test_hidden_rename.py": _d("""
            import pathlib
            import warnings

            import pytest
            from shop.cart import Cart
            from shop.pricing import calc_total, order_total


            def test_new_name_and_discount():
                assert order_total([(10, 2)]) == 20
                assert order_total([(10, 2)], discount=0.25) == 15
                with pytest.raises(ValueError):
                    order_total([(1, 1)], discount=1.5)


            def test_alias_warns():
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    assert calc_total([(1, 3)]) == 3
                assert any(issubclass(x.category, DeprecationWarning) for x in w)


            def test_cart_uses_new_name():
                c = Cart()
                c.add(1.5, 2)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", DeprecationWarning)
                    assert c.total() == 3.0
                src = pathlib.Path("shop/cart.py").read_text()
                assert "order_total" in src and "calc_total" not in src
        """)},
        solution={
            "shop/pricing.py": _d("""
                import warnings


                def order_total(items, discount=0):
                    if not 0 <= discount <= 1:
                        raise ValueError("discount must be between 0 and 1")
                    return round(sum(price * qty for price, qty in items) * (1 - discount), 2)


                def calc_total(items):
                    warnings.warn("calc_total is deprecated, use order_total", DeprecationWarning, stacklevel=2)
                    return order_total(items)
            """),
            "shop/cart.py": _d("""
                from shop.pricing import order_total


                class Cart:
                    def __init__(self):
                        self.items = []

                    def add(self, price, qty=1):
                        self.items.append((price, qty))

                    def total(self):
                        return order_total(self.items)
            """),
        },
        hint_files=["shop/pricing.py", "shop/cart.py"],
    ),
    BenchTask(
        name="word_wrap", level="hard",
        prompt=_d("""
            Implement `wrap(text, width)` in textwrap2.py WITHOUT using the standard textwrap module. Rules:
            - width must be >= 1, else ValueError;
            - words are separated by any whitespace; output lines are joined by single spaces;
            - greedy: put as many words on a line as fit (line length <= width);
            - a word longer than width is split into chunks of exactly `width` characters (the last chunk may be
              shorter) and the remaining words continue after the last chunk if they fit on its line;
            - blank lines in the input (two or more newlines) separate paragraphs; keep one empty line between
              paragraphs in the output;
            - return a list of lines; empty or whitespace-only text returns [].
        """),
        files={
            "textwrap2.py": _d("""
                def wrap(text, width):
                    raise NotImplementedError
            """),
            "tests/test_textwrap2.py": _d("""
                from textwrap2 import wrap


                def test_empty():
                    assert wrap("", 10) == []
            """),
        },
        hidden={"tests/test_hidden_word_wrap.py": _d("""
            import pytest
            from textwrap2 import wrap


            def test_greedy():
                assert wrap("the quick brown fox jumps", 10) == ["the quick", "brown fox", "jumps"]


            def test_exact_fit_and_whitespace():
                assert wrap("  aaa   bbb\\tccc ", 7) == ["aaa bbb", "ccc"]


            def test_long_word_split_and_continue():
                assert wrap("abcdefghij xy z", 4) == ["abcd", "efgh", "ij", "xy z"]
                assert wrap("hi abcdefg", 3) == ["hi", "abc", "def", "g"]


            def test_long_word_last_chunk_shares_line():
                assert wrap("abcde f", 4) == ["abcd", "e f"]


            def test_paragraphs():
                assert wrap("one two\\n\\n\\nthree", 20) == ["one two", "", "three"]


            def test_bad_width():
                with pytest.raises(ValueError):
                    wrap("x", 0)


            def test_blank():
                assert wrap(" \\n \\n", 5) == []
        """)},
        solution={"textwrap2.py": _d("""
            import re


            def _wrap_para(words, width):
                lines, cur = [], ""
                for w in words:
                    while len(w) > width:
                        if cur:
                            lines.append(cur)
                            cur = ""
                        lines.append(w[:width])
                        w = w[width:]
                    if not w:
                        continue
                    if not cur:
                        cur = w
                    elif len(cur) + 1 + len(w) <= width:
                        cur += " " + w
                    else:
                        lines.append(cur)
                        cur = w
                if cur:
                    lines.append(cur)
                return lines


            def wrap(text, width):
                if width < 1:
                    raise ValueError("width must be >= 1")
                paras = [p.split() for p in re.split(r"\\n[ \\t]*\\n", text)]
                paras = [p for p in paras if p]
                out = []
                for i, words in enumerate(paras):
                    if i:
                        out.append("")
                    out += _wrap_para(words, width)
                return out
        """)},
        hint_files=["textwrap2.py"],
    ),
    BenchTask(
        name="roman", level="hard",
        prompt=_d("""
            In roman.py implement `to_roman(n)` for integers 1..3999 (TypeError for non-int, including bool;
            ValueError outside the range) and `from_roman(s)` that accepts only canonical numerals (e.g. "IV" is valid,
            "IIII", "IIV", "VX", "MMMM", "" and lowercase are invalid -> ValueError). from_roman(to_roman(n)) == n for
            every valid n. Add tests.
        """),
        files={
            "roman.py": "",
            "tests/test_roman.py": _d("""
                import roman


                def test_module():
                    assert roman
            """),
        },
        hidden={"tests/test_hidden_roman.py": _d("""
            import pytest
            from roman import from_roman, to_roman


            def test_known():
                assert to_roman(1994) == "MCMXCIV" and to_roman(3999) == "MMMCMXCIX" and to_roman(4) == "IV"
                assert from_roman("MCMXCIV") == 1994


            def test_roundtrip():
                assert all(from_roman(to_roman(n)) == n for n in range(1, 4000))


            @pytest.mark.parametrize("bad", ["IIII", "IIV", "VX", "MMMM", "", "iv", "IC", "XM", "VV", "ABC"])
            def test_invalid(bad):
                with pytest.raises(ValueError):
                    from_roman(bad)


            @pytest.mark.parametrize("bad", [0, 4000, -1])
            def test_range(bad):
                with pytest.raises(ValueError):
                    to_roman(bad)


            @pytest.mark.parametrize("bad", [1.0, "5", True])
            def test_type(bad):
                with pytest.raises(TypeError):
                    to_roman(bad)
        """)},
        solution={"roman.py": _d("""
            _VALS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                     (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


            def to_roman(n):
                if isinstance(n, bool) or not isinstance(n, int):
                    raise TypeError("int required")
                if not 1 <= n <= 3999:
                    raise ValueError("1..3999")
                out = []
                for v, s in _VALS:
                    while n >= v:
                        out.append(s)
                        n -= v
                return "".join(out)


            _TABLE = {to_roman(i): i for i in range(1, 4000)}


            def from_roman(s):
                if s not in _TABLE:
                    raise ValueError(f"not a canonical roman numeral: {s!r}")
                return _TABLE[s]
        """)},
        hint_files=["roman.py"],
    ),
]
