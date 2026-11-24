from dataclasses import dataclass
from html import escape
from typing import Callable
import re

import numpy as np
import pandas as pd
from pandas.api import types as dt


@dataclass(frozen=True)
class Step:
    action: str
    column: int | None
    description: str
    value: object = None


@dataclass(frozen=True)
class Report:
    source: str
    shape: tuple[int, int]
    memory: int
    steps: tuple[Step, ...]
    notes: tuple[str, ...]

    def __str__(self):
        rows, cols = self.shape
        header = ["DATAENGINE  ·  PRE-CLEANING DIAGNOSIS",
                  f"{self.source}  |  {rows:,} rows × {cols:,} columns  |  {self.memory / 2**20:.2f} MiB"]
        details = [f"{i}. {s.description}" for i, s in enumerate(self.steps, 1)]
        return "\n".join([*header, "", *details, *self.notes, "",
                          f"{len(details)} operations proposed. Your input is unchanged."])

    def _repr_html_(self):
        title, summary, *body = str(self).splitlines()
        return (
            '<div style="background:#202820;color:#e4e7d6;border-left:4px solid #ed954e;'
            'padding:1em 1.3em;font-family:monospace">'
            f'<strong style="color:#ffa964">{escape(title)}</strong>'
            f'<div style="margin-top:.4em">{escape(summary)}</div>'
            '<pre style="color:inherit;background:transparent;white-space:pre-wrap">'
            + escape("\n".join(body).strip()) + '</pre></div>'
        )


def _numeric_text(values, label):
    """Accept simple decimal text. Preserve identifier-like and high-precision strings."""
    if re.search(r"(?:^|[^a-z])(id|code|zip|postal|phone|account|sku)(?:$|[^a-z])",
                 str(label).lower()):
        return False
    return all(isinstance(v, str) and len(v.strip()) <= 15
               and re.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", v.strip())
               for v in values)


def _numbers(series):
    return pd.to_numeric(series.replace(r"^\s*$", pd.NA, regex=True), errors="raise").astype("Float64")


def profile(df: pd.DataFrame, *, source="DataFrame") -> Report:
    """Inspect the supplied frame. All proposed values are fixed at review time."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    steps, notes = [], []
    try:
        duplicates = int(df.duplicated().sum())
    except (TypeError, ValueError):
        duplicates = 0
        notes.append("Review duplicates manually · row comparison unavailable for these cell types.")
    if duplicates:
        steps.append(Step("deduplicate", None, f"Duplicates · remove {duplicates:,} repeated rows (keep first)."))

    for pos, label in enumerate(df.columns):
        s = df.iloc[:, pos]
        name = f"{label!r} [column {pos + 1}]"
        observed = s.dropna()
        if isinstance(s.dtype, pd.SparseDtype):
            notes.append(f"{name} · sparse column ({int(s.isna().sum())} missing). Review manually.")
            continue
        text = (len(observed) > 0 and not isinstance(s.dtype, pd.CategoricalDtype)
                and (dt.is_object_dtype(s) or dt.is_string_dtype(s))
                and all(isinstance(v, str) for v in observed))
        if text:
            blanks = s.str.fullmatch(r"\s*", na=False)
            count = int(blanks.sum())
            if count:
                steps.append(Step("blank", pos, f"{name} · treat {count:,} whitespace-only cells as missing."))
                s = s.mask(blanks)
                observed = s.dropna()
            if len(observed) and _numeric_text(observed, label):
                steps.append(Step("numeric", pos, f"{name} · convert numeric text to nullable Float64."))
                s = _numbers(s)

        missing = int(s.isna().sum())
        prefix = f"{name} · {missing:,} missing ({missing / len(s):.1%})" if missing else ""
        numeric = (dt.is_float_dtype(s) or dt.is_integer_dtype(s)) and not dt.is_bool_dtype(s)
        if numeric:
            finite = s.dropna()
            finite = finite[np.isfinite(finite)]
            infinities = len(s) - missing - len(finite)
            if infinities:
                notes.append(f"{name} · {infinities} infinite values require review.")
            # Float-based quantiles can lose integer precision beyond this range.
            if len(finite) and finite.min() >= -(2**53 - 1) and finite.max() <= 2**53 - 1:
                finite = finite.astype("float64")
                median = float(finite.median())
                if missing:
                    steps.append(Step("median", pos, f"{prefix} → fill with median {median:g}.", median))
                if len(finite) >= 8:
                    q1, q3 = finite.quantile([.25, .75])
                    spread = q3 - q1
                    low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
                    outside = int(((finite < low) | (finite > high)).sum())
                    if spread > 0 and outside:
                        steps.append(Step("clip", pos, f"{name} · clip {outside} finite IQR outliers "
                                          f"to [{low:g}, {high:g}].", (low, high)))
                continue
        elif missing and len(s.dropna()) and (text or dt.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)):
            modes = s.mode(dropna=True)
            if len(modes) == 1:
                steps.append(Step("mode", pos, f"{prefix} → fill with mode {modes.iloc[0]!r}.", modes.iloc[0]))
                continue
        if missing:
            notes.append(f"{prefix}. Review manually.")
        elif dt.is_object_dtype(s) and len(observed) and not text:
            notes.append(f"{name} · mixed or non-text object values. Review manually.")
    return Report(str(source), df.shape, int(df.memory_usage(deep=True).sum()), tuple(steps), tuple(notes))


def _apply(df, steps):
    result = df.copy()
    for step in steps:
        if step.action == "deduplicate":
            result = result.drop_duplicates()
            continue
        s = result.iloc[:, step.column]
        if step.action == "blank":
            s = s.mask(s.str.fullmatch(r"\s*", na=False))
        elif step.action == "numeric":
            s = _numbers(s)
        elif step.action == "mode":
            s = s.fillna(step.value)
        else:
            if dt.is_integer_dtype(s) and any(float(v) % 1 for v in
                    (step.value if step.action == "clip" else [step.value])):
                s = s.astype("Float64")
            if step.action == "median":
                s = s.fillna(step.value)
            else:
                s = s.where(~np.isfinite(s), s.clip(*step.value))
        result.isetitem(step.column, s)
    return result


def _choose(report, ask):
    while True:
        choice = ask("Proceed [y], abort [n], or customize [c]? ").strip().lower()
        if choice in ("y", "n"):
            return report.steps if choice == "y" else ()
        if choice != "c":
            print("Enter y, n, or c.")
            continue
        raw = ask("Step numbers separated by commas (Enter to return): ").strip()
        if not raw:
            continue
        try:
            ids = {int(v.strip()) for v in raw.split(",")}
        except ValueError:
            ids = set()
        if not ids or min(ids) < 1 or max(ids) > len(report.steps):
            print("Select numbers from the report.")
            continue
        selected = tuple(s for i, s in enumerate(report.steps, 1) if i in ids)
        # Blanks are prerequisites for conversion, and conversion for numeric operations.
        required = {(s.column, action) for s in selected for action in
                    {"numeric": ["blank"], "median": ["blank", "numeric"],
                     "clip": ["blank", "numeric"], "mode": ["blank"]}.get(s.action, [])}
        missing = [str(i) for i, s in enumerate(report.steps, 1)
                   if (s.column, s.action) in required and i not in ids]
        if missing:
            print("These steps require earlier step(s): " + ", ".join(missing))
            continue
        print("\n".join(s.description for s in selected))
        if ask("Apply these selected steps [y/n]? ").strip().lower() == "y":
            return selected
        return ()


def clean(df: pd.DataFrame, *, source="DataFrame", ask: Callable | None = None) -> pd.DataFrame:
    """Show the blueprint and request approval before applying it to a copy."""
    report = profile(df, source=source)
    try:
        from IPython import get_ipython
        from IPython.display import display
        notebook = getattr(get_ipython(), "kernel", None) is not None
    except ImportError:
        notebook = False
    if notebook:
        display(report)
    else:
        print(report)
    try:
        selected = _choose(report, input if ask is None else ask) if report.steps else ()
    except (KeyboardInterrupt, EOFError):
        selected = ()
    result = _apply(df, selected)
    print(f"Applied {len(selected)} operations. Returned {len(result):,} rows.")
    return result
