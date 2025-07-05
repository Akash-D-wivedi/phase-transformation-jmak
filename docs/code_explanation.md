# Code Explanation

This document walks through the `src/jmak_model.py` code step by step, explaining how each section implements the JMAK fitting pipeline and generates synthetic curves with uncertainty quantification.

---

## 1. Imports & Configuration

```python
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
from pathlib import Path
```

* **NumPy & Pandas** for numerical arrays and dataframes.
* **SciPy** for non-linear least-squares (`least_squares`) and Arrhenius curve fitting (`curve_fit`).
* **scikit-learn GP** for Gaussian‑process regression to smooth log(k) vs T.
* **Matplotlib** for plotting.
* **Pathlib** for robust file-path handling.

```python
# Constants and file paths
R = 8.314                 # Gas constant J/(mol·K)
LOGGRID_N  = 50           # Number of points per synthetic curve
gp_length  = 100.0       # Length scale for GP kernel
CSV_FILE   = Path("data/TTT_data_Steel.csv")
OUT_XLSX   = Path("data/TTT_synthetic.xlsx")
```

* **`R`** used in Arrhenius formula.
* **`LOGGRID_N`** controls time resolution (log scale).
* **`gp_length`** tunes smoothness of Arrhenius interpolation.
* File paths point to **raw CSV** and **output Excel**.

---

## 2. Data Loading & Cleaning

```python
# Auto‑detect encoding and read CSV
def read_csv_auto(path):
    import chardet
    raw = open(path, "rb").read(32768)
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    sep = '\t' if enc.lower().startswith("utf-16") else ','
    return pd.read_csv(path, encoding=enc, sep=sep, engine="python")

df_raw = read_csv_auto(CSV_FILE)
```

* The helper **`read_csv_auto`** uses **chardet** to sniff encoding and chooses comma vs tab separator.

```python
# Normalize column names
def norm(c):
    import re, unicodedata
    s = unicodedata.normalize("NFKD", str(c))
    s = re.sub(r"[^a-z0-9]", "_", s.strip().lower())
    return re.sub(r"_+", "_", s).strip("_")

df_raw.columns = [norm(c) for c in df_raw.columns]
```

* Lowercases, strips spaces and special chars, ensuring consistent headers like `phase`, `t_c`, `t_10`, etc.

```python
# Verify we have required columns
target = {"phase","t_c","t_start","t_10","t_90","y_max"}
have   = set(df_raw.columns)
assert target.issubset(have), f"Missing: {target - have}"
```

---

## 3. Tidy Long Format

```python
# Melt the three anchor times into long form
tidy = (
    df_raw
      .melt(
         id_vars=["phase","t_c","y_max"],
         value_vars=["t_start","t_10","t_90"],
         var_name="time_kind", value_name="time_s"
      )
      .assign(fraction=lambda d: d.time_kind.map({
         "t_start":0.0, "t_10":0.10, "t_90":0.90
      }))
      .dropna(subset=["time_s"])
)
```

* Creates one row per `(phase, T_C, time_kind)` with corresponding `fraction` (0, 0.1, 0.9).
* Simplifies grouping during the JMAK fit.

---

## 4. JMAK Fit: Non-Linear Least Squares in Log-**k**

```python
results = {}
for phase, grp in tidy.groupby("phase"):
    Ts = np.sort(grp["t_c"].unique())

    # Helpers to pack/unpack [n, log(k1), log(k2), ...]
    def pack(n, logk): return np.r_[n, logk]
    def unpack(p):
        n_fit = p[0]
        ks = np.exp(p[1:])
        return n_fit, dict(zip(Ts, ks))

    # Initial guess: n=2, k ~ 1/mean(time)
    k0 = 1 / grp.groupby("t_c")["time_s"].mean().reindex(Ts).values
    p0 = pack(2.0, np.log(np.clip(k0, 1e-12, None)))

    # Residuals in log-k space
    def residuals(p):
        n_fit, kdict = unpack(p)
        pred = grp.apply(
            lambda r: r.y_max * (1 - np.exp(-(kdict[r.t_c]*r.time_s)**n_fit)),
            axis=1
        )
        return pred.values - (grp.fraction * grp.y_max).values

    sol = least_squares(residuals, p0, max_nfev=20000)
    n_fit, kdict = unpack(sol.x)
    results[phase] = {"n": n_fit, "k_T": kdict}
```

1. **Group** by phase.
2. **Pack parameters**: first entry is Avrami exponent `n`; subsequent entries are `log(k)` for each `T`.
3. **Residuals** compare model `y_max*(1 - exp[-(k t)^n])` to the anchor fractions.
4. **`least_squares`** solves for `n` and each `k(T)` simultaneously.

---

## 5. Arrhenius Fit

```python
arr_params = {}
for phase, info in results.items():
    Tvals = np.array(list(info["k_T"].keys()))
    kvals = np.array(list(info["k_T"].values()))

    def arrh(T, k0, Q):
        return k0 * np.exp(-Q/(R*(T+273.15)))

    popt, _ = curve_fit(arrh, Tvals, kvals, p0=[1e6, 80e3])
    k0_fit, Q_fit = popt
    arr_params[phase] = {"k0":k0_fit, "Q":Q_fit}
```

* Uses **`curve_fit`** to extract the Arrhenius parameters $k_0$ and $Q$ for each phase.
* Initial guesses and bounds can be adjusted based on data ranges.

---

## 6. GP Smoothing & Synthetic Curves

```python
from sklearn.gaussian_process import GaussianProcessRegressor

synthetic = []
for phase, info in results.items():
    n_fit = info["n"]
    kdict = info["k_T"]

    # Train GP on log(k) vs T
    T_train = np.array(list(kdict.keys())).reshape(-1,1)
    y_train = np.log(np.array(list(kdict.values())))
    kernel = C(1.0) * RBF(gp_length)
    gp = GaussianProcessRegressor(kernel, alpha=0.05, normalize_y=True)
    gp.fit(T_train, y_train)

    # Generate curves per T
    for T,k in kdict.items():
        tmin = max(1e-3, (0.01/k)**(1/n_fit))
        tmax = tidy.query("phase==@phase and t_c==@T").time_s.max()*5
        tgrid = np.logspace(np.log10(tmin), np.log10(tmax), LOGPOINTS)
        frac = info["y_max"] * (1 - np.exp(-(k*tgrid)**n_fit))
        synthetic += [[phase, T, t, f] for t,f in zip(tgrid, frac)]
```

* **GaussianProcessRegressor** smooths $\ln k$ against temperature.
* Produces a **log-spaced** time grid around each isotherm.
* Computes $f(t)$ on that grid for downstream use.

---

## 7. Visualization & Export

```python
# Plot data vs. JMAK fit + 95% CI
for phase, info in results.items():
    # ... plot original anchor points, fitted curves, and GP-based CI bands ...

# Write to Excel
with pd.ExcelWriter(OUT_XLSX) as writer:
    pd.DataFrame(synthetic, columns=["phase","T_C","time_s","fraction"])\
      .to_excel(writer, sheet_name="synthetic_curves", index=False)
    for phase, info in arr_params.items():
        pd.DataFrame({
            "T_C":list(results[phase]["k_T"].keys()),
            "k_fit":list(results[phase]["k_T"].values()),
            "k0":info["k0"],
            "Q":info["Q"]
        }).to_excel(writer, sheet_name=f"{phase}_kinetics", index=False)
```

* Creates **overlay plots** of raw data and model predictions on a log-time axis.
* Shades **95 % confidence bands** computed via the GP’s predictive standard deviation.
* Exports all synthetic curves and kinetics parameters to a multi-sheet Excel workbook for easy sharing.

---

### Summary

* **Data ingestion**: robust CSV reading + cleaning.
* **Parameter estimation**: JMAK fit for $n,k(T)$; Arrhenius fit for $k_0,Q$.
* **Interpolation**: Gaussian-process smoothing of $\ln k$ vs. $T$.
* **Output**: synthetic curves, model parameters, and publication‐quality figures.

This modular structure lets you easily swap in new data, adjust fitting bounds, or replace the GP with another interpolator. Feel free to explore and extend each step!
