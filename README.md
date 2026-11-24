# DataEngine

Review a cleaning blueprint in Jupyter and apply approved steps to a separate pandas DataFrame. The package accepts a DataFrame from any tabular source pandas can read.

## Run the notebook

From this directory, in your Python environment, run:

```sh
python -m pip install -e '.[notebook]'
python -m jupyterlab DataEngine.ipynb
```

Run the sample cells in order, or pass your table directly:

```python
import pandas as pd
from dataengine import clean, profile

df = pd.read_csv("customers.csv")
profile(df, source="customers.csv")          # preview only
cleaned = clean(df, source="customers.csv")  # review and respond in the cell
```

Choose **y** for all steps, **n** to abort, or **c** to enter comma-separated step numbers. Customization shows the selected steps for confirmation. If a step needs a proposed conversion or blank normalization first, the prompt identifies its number. An empty plan returns a copy immediately. Export the result with pandas if needed.

## What the blueprint means

- Exact duplicate rows can be removed, keeping their first occurrence and index. Duplicate comparison excludes the index.
- Whitespace-only text can be normalized to missing. Simple decimal text can be converted to nullable Float64. Common identifier labels, leading zeros and long numeric strings are protected by heuristics. Review numeric codes and rare but valid extremes before approving.
- Numeric nulls can be filled with the finite median. Text, categorical and Boolean nulls use a unique mode when available. All-null columns, tied modes and unsupported missing-value types require a domain decision.
- With at least eight finite observations and a positive interquartile range, values outside the 1.5 × IQR fences can be clipped. Infinities are reported for review.
- Counts, fill values and bounds describe the supplied frame before row removal, including any proposed blank normalization and numeric interpretation. The number of affected cells can decrease after deduplication. For modeling, this interactive workflow computes statistics on the supplied frame. It does not implement train/test fit and transform.
- Column positions disambiguate repeated labels. Dates, mixed objects and nested values retain their original representation. The engine reports when pandas cannot compare rows for duplicates.

Run the regression suite with `python -m unittest discover -s tests`.
