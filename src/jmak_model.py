"""
src/jmak_model.py

Phase-transformation modeling using the JMAK equation:
- Load raw TTT data from CSV
- Fit Avrami exponent n and per-T rate constants k(T)
- Fit Arrhenius parameters k0 and Q
- Smooth k(T) with a Gaussian Process, generate synthetic f(t) curves
- Export results to Excel and plot fit quality with 95% CI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from pathlib import Path
import chardet, re, unicodedata, sys, warnings

# -----------------------------------------------------------------------------
# 0. CONFIGURATION
# -----------------------------------------------------------------------------
R       = 8.314                  # J/(mol·K)
LOGPOINTS = 50                   # synthetic time points per curve
GP_LS     = 100.0                # GP length-scale (°C)
DATA_DIR  = Path(__file__).parent.parent / "data"
CSV_FILE  = DATA_DIR / "TTT_data_Steel.csv"
OUT_XLSX  = DATA_DIR / "TTT_synthetic.xlsx"

# -----------------------------------------------------------------------------
# 1. DATA LOADING & CLEANING
# -----------------------------------------------------------------------------
def read_csv_auto(path: Path) -> pd.DataFrame:
    raw = open(path, "rb").read(32768)
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    sep = "\t" if enc.lower().startswith("utf-16") else ","
    return pd.read_csv(path, encoding=enc, sep=sep, engine="python")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c):
        s = unicodedata.normalize("NFKD", str(c))
        s = re.sub(r"\s+", "_", s.strip().lower())
        s = re.sub(r"[^a-z0-9_]", "_", s)
        return re.sub(r"_+", "_", s).strip("_").rstrip("_")
    df.columns = [norm(c) for c in df.columns]
    return df

# load and clean
df_raw = read_csv_auto(CSV_FILE)
df_raw = normalize_cols(df_raw)

# rename to expected keys
df_raw = df_raw.rename(columns={
    "phase":"phase",
    "t_c":"t_c",
    "t_start":"t_start",
    "t_10":"t_10",
    "t_90":"t_90",
    "y_max":"y_max"
})

required = {"phase","t_c","t_start","t_10","t_90","y_max"}
if not required.issubset(df_raw.columns):
    missing = required - set(df_raw.columns)
    sys.exit(f"ERROR: missing columns {missing}")

# make sure phase is lowercase
df_raw["phase"] = df_raw["phase"].str.lower()

# -----------------------------------------------------------------------------
# 2. TIDY LONG FORMAT
# -----------------------------------------------------------------------------
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

# build y_max lookup
ymax_map = {
    (row.phase, row.t_c): row.y_max
    for row in df_raw.itertuples()
}

# -----------------------------------------------------------------------------
# 3. JMAK FIT: global n + per-T k in log-space
# -----------------------------------------------------------------------------
results = {}
for phase, grp in tidy.groupby("phase"):
    Ts = np.sort(grp["t_c"].unique())

    # pack/unpack (n, logk1..logkM)
    def pack(n, logk): return np.r_[n, logk]
    def unpack(p):
        n_fit = p[0]
        ks = np.exp(p[1:])
        return n_fit, dict(zip(Ts, ks))

    # initial guess
    k0 = 1.0 / grp.groupby("t_c")["time_s"].mean().reindex(Ts).values
    p0 = pack(2.0, np.log(np.clip(k0, 1e-12, None)))

    # residuals
    def resid(p):
        n_fit, kdict = unpack(p)
        pred = grp.apply(
            lambda r: r.y_max*(1 - np.exp(-(kdict[r.t_c]*r.time_s)**n_fit)),
            axis=1
        )
        return pred.values - (grp.fraction * grp.y_max).values

    sol = least_squares(resid, p0, max_nfev=20000)
    if not sol.success:
        warnings.warn(f"[{phase}] JMAK fit did not converge: {sol.message}")
    n_fit, kdict = unpack(sol.x)
    results[phase] = {"n":n_fit, "k_T":kdict}

    print(f"{phase:8s} → n = {n_fit:.2f}, k range = "
          f"{min(kdict.values()):.2e}–{max(kdict.values()):.2e} 1/s")

# -----------------------------------------------------------------------------
# 4. ARRHENIUS ANALYSIS: k0 and Q for each phase
# -----------------------------------------------------------------------------
arr_params = {}
for phase, info in results.items():
    kdf = pd.DataFrame({
        "T": list(info["k_T"].keys()),
        "k": list(info["k_T"].values())
    })
    Tvals, kvals = kdf["T"].values, kdf["k"].values

    def arrh(T, k0, Q):
        return k0 * np.exp(-Q/(R*(T+273.15)))

    popt, _ = curve_fit(
        arrh, Tvals, kvals,
        p0=[1e6, 8e4],
        bounds=([0,0],[np.inf,np.inf])
    )
    k0_fit, Q_fit = popt
    arr_params[phase] = {"k0":k0_fit, "Q":Q_fit}
arr_df = pd.DataFrame([
    {"phase":ph, "k0 (1/s)":p["k0"], "Q (kJ/mol)":p["Q"]/1000.0}
    for ph,p in arr_params.items()
])
print("\nArrhenius parameters:")
print(arr_df.to_string(index=False))

# -----------------------------------------------------------------------------
# 5. GP SMOOTHING & SYNTHETIC CURVE GENERATION
# -----------------------------------------------------------------------------
synthetic = []
for phase, info in results.items():
    n_fit = info["n"]
    kdict = info["k_T"]

    # train GP on ln k vs T
    T_arr = np.array(list(kdict.keys())).reshape(-1,1)
    y_arr = np.log(list(kdict.values()))
    kernel = C(1.0, (1e-3,1e3)) * RBF(GP_LS, (10,1000))
    gp = GaussianProcessRegressor(kernel, alpha=0.05, normalize_y=True)
    gp.fit(T_arr, y_arr)

    # generate curves at each T
    for T,k in kdict.items():
        ymax = ymax_map[(phase, T)]
        tmin = max(1e-3, (0.01/k)**(1/n_fit))
        tmax = tidy.query("phase==@phase and t_c==@T").time_s.max()*5
        tgrid = np.logspace(np.log10(tmin), np.log10(tmax), LOGPOINTS)
        frac  = ymax*(1 - np.exp(-(k*tgrid)**n_fit))
        for t,f in zip(tgrid, frac):
            synthetic.append({"phase":phase, "T_C":T, "time_s":t, "fraction":f})

syn_df = pd.DataFrame(synthetic)

# -----------------------------------------------------------------------------
# 6. EXPORT TO EXCEL
# -----------------------------------------------------------------------------
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
    syn_df.to_excel(w, sheet_name="synthetic_curves", index=False)
    for phase,p in arr_params.items():
        kd = results[phase]["k_T"]
        pd.DataFrame({
            "T_C": list(kd.keys()),
            "k_fit": list(kd.values()),
            "k0": p["k0"],
            "Q": p["Q"]
        }).to_excel(w, sheet_name=f"{phase}_kinetics", index=False)

print(f"\n Written synthetic data + kinetics to {OUT_XLSX}")

# -----------------------------------------------------------------------------
# 7. OPTIONAL PLOTTING (uncomment to view)
# -----------------------------------------------------------------------------
# for phase, info in results.items():
#     n_fit = info["n"]
#     kdict = info["k_T"]
#     kdf   = pd.DataFrame({"T":list(kdict),"k":list(kdict.values())})
#     gp    = GaussianProcessRegressor(C(1.0)*RBF(GP_LS), alpha=0.05, normalize_y=True)\
#                .fit(kdf["T"].values.reshape(-1,1), np.log(kdf["k"]))
#
#     fig, ax = plt.subplots()
#     o = tidy[tidy.phase==phase]
#     s = syn_df[syn_df.phase==phase]
#     Ts = sorted(o["T_C"].unique())
#     Ts_choice = [Ts[0], Ts[len(Ts)//2], Ts[-1]]
#
#     for T in Ts_choice:
#         pts = o[o.T_C==T]
#         ax.scatter(pts.time_s, pts.fraction, label=f"{T}°C data")
#         curve = s[s.T_C==T]
#         ax.plot(curve.time_s, curve.fraction, label=f"{T}°C fit")
#
#         ymax = ymax_map[(phase,T)]
#         mu,sig = gp.predict([[T]], return_std=True)
#         k_hi = np.exp(mu+1.96*sig)[0]; k_lo=np.exp(mu-1.96*sig)[0]
#         tvals=curve.time_s.values
#         f_hi = ymax*(1-np.exp(-(k_hi*tvals)**n_fit))
#         f_lo = ymax*(1-np.exp(-(k_lo*tvals)**n_fit))
#         ax.fill_between(tvals,f_lo,f_hi,color="gray",alpha=0.3)
#
#     ax.set_xscale("log"); ax.set_ylabel("Fraction"); ax.set_xlabel("Time (s)")
#     ax.set_title(f"{phase.capitalize()} JMAK + CI"); ax.legend(); plt.show()
