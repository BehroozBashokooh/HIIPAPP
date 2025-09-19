import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats as stt
from scipy.optimize import minimize
from numpy.linalg import cholesky, eigh
from datetime import datetime
import io, zipfile

# =============================================================
# Config
# =============================================================
APP_NAME = "HIIP APP, Hurrah!"
APP_VER = "v1.1.5"
st.set_page_config(page_title=f"{APP_NAME} — {APP_VER}", layout="centered")

# Inject custom styles once for the volumetric formula block
if "_vol_formula_style" not in st.session_state:
    st.markdown(
        """
        <style>
        .volumetric-formula {
            border-left: 4px solid #4c8bf5;
            background-color: rgba(76, 139, 245, 0.08);
            padding: 0.75rem 1rem;
            margin: 0 0 1rem 0;
            font-size: 0.95rem;
            line-height: 1.55;
            border-radius: 0 6px 6px 0;
        }
        .volumetric-formula .label {
            font-weight: 600;
            color: #1f2933;
        }
        .volumetric-formula .optional {
            font-weight: 600;
            color: #0f4c81;
        }
        .volumetric-formula .math-symbol {
            font-family: "Menlo", "DejaVu Sans Mono", monospace;
            font-size: 0.9rem;
            margin: 0 0.05rem;
        }
        .volumetric-formula sub { font-size: 0.8em; }
        .volumetric-formula sup { font-size: 0.8em; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_vol_formula_style"] = True

# =============================================================
# Input helpers
# =============================================================
def float_text_input(label, default, key, min_value=None, max_value=None, fmt="%.6f", help=None):
    """Locale-agnostic float input via text field.

    Accepts both '.' and ',' as decimal separators and tolerates partial typing.
    Returns the last valid parsed float; shows the raw text as typed.
    """
    raw_key = f"{key}__raw"
    val_key = f"{key}__val"

    # Initialize defaults on first run
    if raw_key not in st.session_state:
        try:
            st.session_state[raw_key] = (fmt % float(default))
        except Exception:
            st.session_state[raw_key] = str(default)
    if val_key not in st.session_state:
        try:
            st.session_state[val_key] = float(default)
        except Exception:
            st.session_state[val_key] = 0.0

    raw = st.text_input(label, value=st.session_state[raw_key], key=raw_key, help=help)

    # Try to parse with tolerant rules
    s = (raw or "").strip().replace("\u200b", "").replace(",", ".")
    last_valid = st.session_state[val_key]
    try:
        v = float(s)
        if min_value is not None and v < min_value:
            st.caption(f"Note: enforcing min = {min_value}")
            v = float(min_value)
        if max_value is not None and v > max_value:
            st.caption(f"Note: enforcing max = {max_value}")
            v = float(max_value)
        st.session_state[val_key] = v
        return v
    except Exception:
        # Keep last valid value while user is still typing
        return last_valid

# =============================================================
# Units and conversions
# =============================================================
AREA_UNITS = {"m²": 1.0, "km²": 1e6, "acres": 4046.8564224}
THICKNESS_UNITS = {"m": 1.0, "ft": 0.3048}
# Oil units: conversion factors from m³ to various oil units
OIL_UNITS = {
    "m³": 1.0, "Mm³": 1e-3, "MMm³": 1e-6,  # Metric volume units (m³ base)
    "stb": 6.2898, "Mstb": 6.2898e-3, "MMstb": 6.2898e-6  # Oil field units (1 m³ = 6.2898 stb)
}
# Gas units: conversion factors from m³ to various gas units
GAS_UNITS = {
    "m³": 1.0, "Mm³": 1e-3, "MMm³": 1e-6,  # Metric volume units (m³ base)
    "scf": 35.3147, "MMscf": 35.3147e-6, "Bscf": 35.3147e-9, "Tscf": 35.3147e-12  # 1 m³ = 35.3147 scf
}

# =============================================================
# Distributions
# =============================================================
def pert_to_beta_params(xmin, mode, xmax, lam=4.0):
    """Stable PERT→Beta mapping."""
    if not (np.isfinite(xmin) and np.isfinite(mode) and np.isfinite(xmax)):
        raise ValueError("PERT parameters must be finite numbers")
    if not (xmin < mode < xmax):
        raise ValueError("PERT requires min < mode < max")
    if lam <= 0:
        raise ValueError("PERT lambda must be > 0")
    span = xmax - xmin
    if span <= 0:
        raise ValueError("PERT requires max > min")
    a = 1.0 + lam * (mode - xmin) / span
    b = 1.0 + lam * (xmax - mode) / span
    eps = 1e-9
    return max(a, eps), max(b, eps)

def trunc_lognorm_build(median, sigma_ln, xmin, xmax):
    """Return (dist_like, ppf, pdf) for a lognormal truncated to [xmin, xmax]."""
    if not (xmax > xmin > 0):
        raise ValueError("Truncated Lognormal requires 0 < min < max")
    if median <= 0 or sigma_ln <= 0:
        raise ValueError("median>0 and sigma_ln>0 required")
    m = np.log(median); s = float(sigma_ln)
    a = (np.log(xmin) - m) / s
    b = (np.log(xmax) - m) / s
    tn = stt.truncnorm(a=a, b=b, loc=m, scale=s)  # in log space
    def ppf(u):
        u = np.asarray(u)
        return np.exp(tn.ppf(u))
    def pdf(x):
        x = np.asarray(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = stt.truncnorm.pdf(np.log(x), a=a, b=b, loc=m, scale=s) * (1.0 / x)
        val[(x < xmin) | (x > xmax)] = 0.0
        return val
    class _TLN:
        def ppf(self_inner, u): return ppf(u)
        def pdf(self_inner, x): return pdf(x)
    return _TLN(), ppf, pdf

def make_distribution(dist_name, params):
    """
    Build a distribution interface: returns (dist_like, ppf, pdf).
    """
    name = dist_name.lower()

    # ----- Uniform (random) -----
    if name in ("uniform (random)", "uniform", "random"):
        lo, hi = params["min"], params["max"]
        if not (hi > lo):
            raise ValueError("Uniform requires max > min")
        dist = stt.uniform(loc=lo, scale=hi - lo)
        return dist, dist.ppf, dist.pdf

    # ----- Triangular -----
    if name == "triangular":
        lo, mode, hi = params["min"], params["mode"], params["max"]
        if not (lo < mode < hi):
            raise ValueError("Triangular requires min < mode < max")
        c = (mode - lo) / (hi - lo)
        dist = stt.triang(c=c, loc=lo, scale=hi - lo)
        return dist, dist.ppf, dist.pdf

    # ----- PERT (scaled Beta) -----
    if name == "pert":
        lo, mode, hi = params["min"], params["mode"], params["max"]
        lam = params.get("lambda", 4.0)
        a, b = pert_to_beta_params(lo, mode, hi, lam)
        dist = stt.beta(a, b, loc=lo, scale=hi - lo)
        return dist, dist.ppf, dist.pdf

    # ----- Normal (optionally truncated) -----
    if name == "normal":
        mu, sigma = params["mean"], params["sd"]
        lo = params.get("min", -np.inf)
        hi = params.get("max", np.inf)
        if sigma <= 0:
            raise ValueError("Normal sd must be > 0")
        if np.isfinite(lo) or np.isfinite(hi):
            a, b = (lo - mu) / sigma, (hi - mu) / sigma
            dist = stt.truncnorm(a=a, b=b, loc=mu, scale=sigma)
        else:
            dist = stt.norm(loc=mu, scale=sigma)
        return dist, dist.ppf, dist.pdf

    # ----- Lognormal -----
    if name == "lognormal":
        med = params["median"]; s = params["sigma_ln"]
        if s <= 0 or med <= 0:
            raise ValueError("Lognormal requires median>0 and sigma_ln>0")
        lo = params.get("min", None)
        hi = params.get("max", None)
        if (lo is not None) or (hi is not None):
            if lo is None or hi is None:
                raise ValueError("Provide both min and max to truncate the Lognormal.")
            dist, ppf, pdf = trunc_lognorm_build(med, s, float(lo), float(hi))
            return dist, ppf, pdf
        else:
            dist = stt.lognorm(s=s, scale=np.exp(np.log(med)))
            return dist, dist.ppf, dist.pdf

    # (Truncated Lognormal handled via 'lognormal' with optional bounds)

    # ----- Beta (scaled to [min,max]) -----
    if name == "beta":
        a, b = params["alpha"], params["beta"]
        lo = params.get("min", 0.0); hi = params.get("max", 1.0)
        if not (a > 0 and b > 0 and hi > lo):
            raise ValueError("Beta requires alpha>0, beta>0, and max>min")
        dist = stt.beta(a, b, loc=lo, scale=hi - lo)
        return dist, dist.ppf, dist.pdf

    # ----- Custom by P10/P50/P90 (exceedance: P10 high, P90 low) -----
    if name == "custom (p10/p50/p90)":
        family = params["family"].lower()
        p10, p50, p90 = params["p10"], params["p50"], params["p90"]
        if not (p90 <= p50 <= p10):
            raise ValueError("Require P90 ≤ P50 ≤ P10 (exceedance: P10 high, P90 low).")
        z10, z90 = stt.norm.ppf(0.10), stt.norm.ppf(0.90)

        if family == "normal":
            mu = p50
            sigma = (p10 - p90) / (z90 - z10)
            if sigma <= 0 or not np.isfinite(sigma):
                raise ValueError("Derived sd ≤ 0; check P10/P50/P90.")
            dist = stt.norm(loc=mu, scale=sigma)
            params["derived"] = {"mean": mu, "sd": sigma, "support": "(-∞, ∞)"}
            return dist, dist.ppf, dist.pdf

        if family == "lognormal":
            if p10 <= 0 or p50 <= 0 or p90 <= 0:
                raise ValueError("Lognormal requires positive P10/P50/P90")
            m = np.log(p50)
            s = (np.log(p10) - np.log(p90)) / (z90 - z10)
            if s <= 0 or not np.isfinite(s):
                raise ValueError("Derived sigma_ln ≤ 0; check quantiles.")
            dist = stt.lognorm(s=s, scale=np.exp(m))
            params["derived"] = {"median": p50, "sigma_ln": s, "support": "(0, ∞)"}
            return dist, dist.ppf, dist.pdf

        if family == "beta":
            lo = params.get("min", 0.0); hi = params.get("max", 1.0)
            if not (hi > lo):
                raise ValueError("For Beta, need max > min")
            y10 = (p10 - lo) / (hi - lo)
            y50 = (p50 - lo) / (hi - lo)
            y90 = (p90 - lo) / (hi - lo)
            if not (0 <= y90 <= y50 <= y10 <= 1):
                raise ValueError("P90 ≤ P50 ≤ P10 must lie within [min,max] for Beta")
            def obj(x):
                a, b = np.exp(x[0]), np.exp(x[1])
                q10 = stt.beta.ppf(0.10, a, b)
                q50 = stt.beta.ppf(0.50, a, b)
                q90 = stt.beta.ppf(0.90, a, b)
                return (q90 - y10)**2 + (q50 - y50)**2 + (q10 - y90)**2
            res = minimize(obj, x0=[np.log(2.0), np.log(2.0)], method="Nelder-Mead", options={"maxiter": 5000})
            a_hat, b_hat = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
            if not np.isfinite(a_hat) or not np.isfinite(b_hat):
                raise ValueError("Could not fit Beta to quantiles.")
            dist = stt.beta(a_hat, b_hat, loc=lo, scale=hi - lo)
            params["derived"] = {"alpha": a_hat, "beta": b_hat, "min": lo, "max": hi, "support": f"[{lo}, {hi}]"}
            return dist, dist.ppf, dist.pdf

        raise ValueError("Unsupported custom family. Choose Normal, Lognormal, or Beta.")

    # ----- Discrete (finite support) -----
    if name == "discrete":
        values = params.get("values", None)
        probs = params.get("probs", None)
        if values is None or probs is None:
            raise ValueError("Discrete requires 'values' and 'probs'.")
        values = np.asarray(values, dtype=float)
        probs = np.asarray(probs, dtype=float)
        if len(values) != len(probs) or len(values) == 0:
            raise ValueError("Discrete: values and probs must have same non-zero length.")
        if np.any(probs < 0):
            raise ValueError("Discrete: negative probabilities not allowed.")
        s = probs.sum()
        if s <= 0:
            raise ValueError("Discrete: probability weights must sum > 0.")
        p = probs / s
        c = np.cumsum(p)
        c[-1] = 1.0
        def ppf(u):
            u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
            idx = np.searchsorted(c, u, side='right')
            idx[idx == len(values)] = len(values)-1
            return values[idx]
        def pdf(x):
            x = np.asarray(x, dtype=float)
            return np.zeros_like(x, dtype=float)
        class _Disc:
            def ppf(self_inner, u): return ppf(u)
            def pdf(self_inner, x): return pdf(x)
        return _Disc(), ppf, pdf

    raise ValueError(f"Unsupported distribution: {dist_name}")

# =============================================================
# Dependence & sampling (LHS + Gaussian copula defaults)
# =============================================================
def nearest_pd(A):
    B = (A + A.T) / 2
    eigvals, eigvecs = eigh(B)
    eigvals[eigvals < 0] = 0
    Bpd = (eigvecs @ np.diag(eigvals) @ eigvecs.T)
    return (Bpd + Bpd.T) / 2

def latin_hypercube(n, k, rng):
    cut = (np.arange(n)[:, None] + rng.random(size=(n, k))) / n
    U = np.zeros((n, k))
    for j in range(k):
        order = rng.permutation(n)
        U[:, j] = cut[order, j]
    return U

def gaussian_copula_sample(n, corr, ppfs, rng, lhs_uniforms=None):
    k = len(ppfs)
    if lhs_uniforms is None:
        Z = rng.normal(size=(n, k))
    else:
        Z = stt.norm.ppf(lhs_uniforms)
    L = cholesky(corr)
    Zc = Z @ L.T
    U = stt.norm.cdf(Zc)
    X = np.column_stack([ppfs[j](U[:, j]) for j in range(k)])
    return X

# =============================================================
# Curve-mode helpers (GRV from A(z))
# =============================================================
def _read_table(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded)
    else:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_excel(uploaded)
    return df

def _normalize_columns(df):
    cols = {c.lower().strip(): c for c in df.columns}
    depth_key = None
    for key in ["depth", "z", "tvd", "tvdss", "md"]:
        if key in cols:
            depth_key = cols[key]; break
    area_key = None
    for key in ["area", "a"]:
        if key in cols:
            area_key = cols[key]; break
    if depth_key is None or area_key is None:
        raise ValueError("Curve file must have columns named like 'depth' and 'area'.")
    df2 = df[[depth_key, area_key]].dropna()
    df2.columns = ["depth", "area"]
    df2 = df2.sort_values("depth")
    return df2

def integrate_grv_m3(depth_m, area_m2, z_top_m, z_base_m):
    """Trapz integrate area vs depth between the two bounds, including interpolated endpoints."""
    if z_base_m < z_top_m:
        z_top_m, z_base_m = z_base_m, z_top_m
    zmin, zmax = depth_m.min(), depth_m.max()
    z0 = np.clip(z_top_m, zmin, zmax)
    z1 = np.clip(z_base_m, zmin, zmax)
    if z1 <= z0:
        return 0.0
    mask = (depth_m >= z0) & (depth_m <= z1)
    z = depth_m[mask]; a = area_m2[mask]
    if len(z) == 0 or z[0] > z0:
        a0 = np.interp(z0, depth_m, area_m2)
        z = np.insert(z, 0, z0); a = np.insert(a, 0, a0)
    elif z[0] < z0:
        a[0] = np.interp(z0, depth_m, area_m2); z[0] = z0
    if z[-1] < z1:
        a1 = np.interp(z1, depth_m, area_m2)
        z = np.append(z, z1); a = np.append(a, a1)
    elif z[-1] > z1:
        a[-1] = np.interp(z1, depth_m, area_m2); z[-1] = z1
    return float(np.trapz(a, x=z))

# =============================================================
# Business logic
# =============================================================
def compute_volumes(df, fluid, include_rf=True):
    """
    Computes in-place and optional derived volumes.
    Inputs expected in df (subset depending on mode):
      - geometry: either GRV_m3, or A_m2 & h_m
      - rock & fluids: NTG, phi, Sw, Bo_rm3_per_stb (oil) or Bg_rm3_per_scf (gas)
      - optional: RF (fraction), Rs (scf/stb for oil), CGR (STB/MMscf for gas)
    Outputs returned (subset depending on inputs):
      - Oil: STOIIP_m3, STOIIP_stb, (optional) Reserves_m3/Reserves_stb, (optional) SGIIP_scf/SGIIP_m3
      - Gas: GIIP_m3, GIIP_scf, (optional) Recoverable_m3/Recoverable_scf, (optional) Condensate_stb/Condensate_m3
    """
    Sh = 1.0 - df["Sw"]
    if "GRV_m3" in df.columns:
        hcpv_m3 = df["GRV_m3"] * df["NTG"] * df["phi"] * Sh
    else:
        hnet = df["h_m"] * df["NTG"]
        hcpv_m3 = df["A_m2"] * hnet * df["phi"] * Sh

    out = {}

    if fluid == "Oil":
        # In-place oil volumes (at standard conditions via Bo)
        stoiip_m3_std = hcpv_m3 / df["Bo_rm3_per_stb"]
        out["STOIIP_m3"]  = stoiip_m3_std
        out["STOIIP_stb"] = stoiip_m3_std * OIL_UNITS["stb"]

        # Optional: Solution Gas Initially In Place (SGIIP) if Rs is provided (scf/stb)
        if "Rs" in df.columns:
            sgiip_scf = out["STOIIP_stb"] * df["Rs"]
            out["SGIIP_scf"] = sgiip_scf
            out["SGIIP_m3"]  = sgiip_scf / GAS_UNITS["scf"]

        # Optional: recoverable oil (Reserves) if RF provided/selected
        if include_rf and "RF" in df.columns:
            out["Reserves_m3"]  = out["STOIIP_m3"]  * df["RF"]
            out["Reserves_stb"] = out["STOIIP_stb"] * df["RF"]

    else:  # Gas
        giip_m3_std = hcpv_m3 / df["Bg_rm3_per_scf"]
        out["GIIP_m3"] = giip_m3_std
        out["GIIP_scf"] = giip_m3_std * GAS_UNITS["scf"]

        # Optional: Condensate Initially In Place (CIP) if CGR is provided (STB/MMscf)
        if "CGR" in df.columns:
            condensate_stb = (out["GIIP_scf"] / 1e6) * df["CGR"]
            out["Condensate_stb"] = condensate_stb
            out["Condensate_m3"]  = condensate_stb / OIL_UNITS["stb"]

        # Optional: recoverable gas if RF provided/selected
        if include_rf and "RF" in df.columns:
            out["Recoverable_m3"] = out["GIIP_m3"] * df["RF"]
            out["Recoverable_scf"] = out["GIIP_scf"] * df["RF"]

    return pd.DataFrame(out)
# =============================================================
# UI — Sidebar
# =============================================================
with st.sidebar:
    st.header(f"{APP_NAME}")
    st.caption(APP_VER)

    st.header("Simulation Settings")
    n = st.number_input("Iterations", min_value=1000, max_value=1_000_000, value=10_000, step=1000, key="n_iter")
    st.caption(f"We will run **{int(n):,}** iterations when you click Run.")
    seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="seed")
    #enable_deps = st.checkbox("Enable dependencies (correlation)", value=True, key="enable_deps")

    st.header("Units")
    u_area = st.selectbox("Area units", list(AREA_UNITS.keys()), index=1, key="u_area")
    u_thick = st.selectbox("Thickness units", list(THICKNESS_UNITS.keys()), index=0, key="u_thick")
    fluid = st.radio("Fluid system", ["Oil", "Gas"], index=0, key="fluid")

    st.header("Output scale & formatting")
    if fluid == "Oil":
        out_unit = st.selectbox("In-place & reserves unit", list(OIL_UNITS.keys()), index=5, key="out_unit")  # default MMstb
    else:
        out_unit = st.selectbox("In-place & recoverable unit", list(GAS_UNITS.keys()), index=4, key="out_unit")  # default MMscf
    decimals = st.number_input("Round results to # decimals", min_value=0, max_value=6, value=2, step=1, key="decimals")
    heatmap_decimals = st.number_input("Heatmap label decimals", min_value=0, max_value=6, value=1, step=1, key="heatmap_decimals")

    # Conditional unit pickers for Solution Gas / Condensate
    if fluid == "Oil" and st.session_state.get("include_rs", False):
        sg_unit = st.selectbox(
            "Solution gas unit",
            ["scf", "MMscf", "Bscf", "Tscf", "m³", "Mm³", "MMm³"],
            index=1,
            key="sg_unit",
            help="Units used when Rs is enabled and SGIIP is computed"
        )
        sg_decimals = st.number_input("Solution gas decimals", min_value=0, max_value=6, value=2, step=1, key="sg_decimals")
    if fluid == "Gas" and st.session_state.get("include_cgr", False):
        cnd_unit = st.selectbox(
            "Condensate unit",
            ["stb", "Mstb", "MMstb", "m³", "Mm³", "MMm³"],
            index=1,
            key="cnd_unit",
            help="Units used when CGR is enabled and Condensate in place is computed"
        )
        cnd_decimals = st.number_input("Condensate decimals", min_value=0, max_value=6, value=2, step=1, key="cnd_decimals")

    st.header("Colors")
    plot_color = st.color_picker("Plot color", value="#1f77b4", key="plot_color")
    annot_color = st.color_picker("Annotation color (P10/P50/P90)", value="#d62728", key="annot_color")

    st.divider()
st.sidebar.markdown("[Click here to send questions or comments](mailto:behrooz.bashokooh@shell.com)")
# =============================================================
# Tabs
# =============================================================
sim_tab, help_tab = st.tabs(["Simulator", "Help / User Guide"])

with sim_tab:
    st.title("Reservoir in-place volumes")
    st.caption(" Monte Carlo with dependencies (LHS + Gaussian copula)")

    if fluid == "Oil":
        volumetric_formula = """
        <div class="volumetric-formula">
            <p><span class="label">STOIIP</span> = (GRV × NTG × &phi; × (1 − S<sub>w</sub>)) ÷ B<sub>o</sub></p>
            <p><span class="optional">* Reserves</span> = STOIIP × RF,&nbsp; <span class="optional">* SGIIP</span> = STOIIP<sub>STB</sub> × R<sub>s</sub></p>
        </div>
        """
    else:
        volumetric_formula = """
        <div class="volumetric-formula">
            <p><span class="label">GIIP</span> = (GRV × NTG × &phi; × (1 − S<sub>w</sub>)) ÷ B<sub>g</sub></p>
            <p><span class="optional">* Recoverable</span> = GIIP × RF,&nbsp; <span class="optional">* Condensate IP</span> = GIIP<sub>scf</sub> ÷ 10<sup>6</sup> × CGR</p>
        </div>
        """

    st.markdown(volumetric_formula, unsafe_allow_html=True)

    st.subheader("GRV Source")
    grv_source = st.radio(
        "Choose how to obtain reservoir volume (GRV)",
        [
            "Uncertain Area × Gross thickness",
            "Upload Area–Depth curve (integrate over full range)",
            "Direct GRV distribution (m³)",
        ],
        index=0,
        key="grv_source"
    )

    rng = np.random.default_rng(int(seed))

    # Prepare holders to avoid NameError in any branch
    input_figs = {}
    curve_preview_fig = None
    A_fig = h_fig = NTG_fig = phi_fig = Sw_fig = Bo_fig = Bg_fig = RF_fig = grv_scale_fig = GRV_fig = None
    Rs_fig = CGR_fig = None

    # ---------- common dist editor ----------
    def dist_editor(label, dist_default, param_defaults, help_text=None, units_label=""):
        with st.expander(label + (f"  [{units_label}]" if units_label else ""), expanded=False):
            col1, col2 = st.columns([1.1, 3])
            with col1:
                dist_name = st.selectbox(
                    "Distribution",
                    [
                        "Uniform (random)",
                        "Triangular",
                        "PERT",
                        "Normal",
                        "Lognormal",
                        "Beta",
                        "Custom (P10/P50/P90)",
                        "Discrete",
                    ],
                    index=dist_default,
                    key=f"dist_{label}",
                )
            with col2:
                params = {}
                try:
                    if dist_name == "Uniform (random)":
                        c1, c2 = st.columns(2)
                        params["min"] = c1.number_input("min", value=param_defaults.get("min", 0.0), key=f"{label}_umin")
                        params["max"] = c2.number_input("max", value=param_defaults.get("max", 1.0), key=f"{label}_umax")

                    elif dist_name in ("Triangular", "PERT"):
                        c1, c2, c3 = st.columns(3)
                        params["min"] = c1.number_input("min", value=param_defaults.get("min", 0.0), key=f"{label}_min")
                        params["mode"] = c2.number_input("mode", value=param_defaults.get("mode", 0.0), key=f"{label}_mode")
                        params["max"] = c3.number_input("max", value=param_defaults.get("max", 1.0), key=f"{label}_max")
                        if dist_name == "PERT":
                            params["lambda"] = st.number_input(
                                "lambda (shape)", value=float(param_defaults.get("lambda", 4.0)),
                                min_value=0.0001, key=f"{label}_lam"
                            )

                    elif dist_name == "Normal":
                        # Choose formatting precision based on magnitude (Bg needs more precision)
                        mean_def = float(param_defaults.get("mean", 0.0))
                        sd_def = float(param_defaults.get("sd", 1.0))
                        min_def = float(param_defaults.get("min", mean_def)) if param_defaults.get("min", None) is not None else mean_def
                        max_def = float(param_defaults.get("max", mean_def)) if param_defaults.get("max", None) is not None else mean_def
                        scale_mag = max(abs(mean_def), abs(sd_def), abs(min_def), abs(max_def))
                        fmt_norm = "%.4f" if scale_mag < 0.1 else "%.2f"

                        c1, c2 = st.columns(2)
                        params["mean"] = float_text_input(
                            "mean",
                            default=mean_def,
                            key=f"{label}_mean",
                            fmt=fmt_norm,
                            help="Accepts '.' or ',' as decimal",
                        )
                        params["sd"] = float_text_input(
                            "sd",
                            default=sd_def,
                            key=f"{label}_sd",
                            min_value=1e-12,
                            fmt=fmt_norm,
                            help="Accepts '.' or ',' as decimal (sd>0)",
                        )

                        # Optional truncation bounds to avoid inf defaults causing UI quirks
                        _nb_min = param_defaults.get("min", None)
                        _nb_max = param_defaults.get("max", None)
                        _nb_has_bounds = (
                            (_nb_min is not None) and (_nb_max is not None)
                            and np.isfinite(float(_nb_min)) and np.isfinite(float(_nb_max))
                        )
                        add_bounds = st.checkbox(
                            "Add bounds (truncate)", value=bool(_nb_has_bounds), key=f"{label}_normal_bounds"
                        )
                        if add_bounds:
                            c3, c4 = st.columns(2)
                            params["min"] = float_text_input(
                                "min",
                                default=float(param_defaults.get("min", -1.0)),
                                key=f"{label}_nmin",
                                fmt=fmt_norm,
                                help="Accepts '.' or ',' as decimal",
                            )
                            params["max"] = float_text_input(
                                "max",
                                default=float(param_defaults.get("max", 1.0)),
                                key=f"{label}_nmax",
                                fmt=fmt_norm,
                                help="Accepts '.' or ',' as decimal",
                            )

                    
                    elif dist_name == "Lognormal":
                        c1, c2 = st.columns(2)
                        params["median"] = float_text_input(
                            "median",
                            default=float(param_defaults.get("median", 1.0)),
                            key=f"{label}_med",
                            min_value=1e-12,
                            fmt="%.2f",
                            help="Accepts '.' or ',' as decimal (median>0)",
                        )
                        params["sigma_ln"] = float_text_input(
                            "sigma_ln (ln-space sd)",
                            default=float(param_defaults.get("sigma_ln", 0.5)),
                            key=f"{label}_sig",
                            min_value=1e-6,
                            fmt="%.2f",
                            help="Accepts '.' or ',' as decimal (sd>0)",
                        )
                        _lnb_min = param_defaults.get("min", None)
                        _lnb_max = param_defaults.get("max", None)
                        _lnb_has_bounds = (
                            (_lnb_min is not None) and (_lnb_max is not None)
                            and np.isfinite(float(_lnb_min)) and np.isfinite(float(_lnb_max))
                        )
                        add_bounds_ln = st.checkbox(
                            "Add bounds (truncate)", value=bool(_lnb_has_bounds), key=f"{label}_ln_bounds"
                        )
                        if add_bounds_ln:
                            d1, d2 = st.columns(2)
                            params["min"] = d1.number_input(
                                "min",
                                value=float(param_defaults.get("min", 0.1)),
                                min_value=1e-12,
                                step=0.01,
                                format="%.2f",
                                key=f"{label}_ln_min",
                            )
                            params["max"] = d2.number_input(
                                "max",
                                value=float(param_defaults.get("max", 10.0)),
                                min_value=1e-12,
                                step=0.01,
                                format="%.2f",
                                key=f"{label}_ln_max",
                            )

                    elif dist_name == "Beta":
                        c1, c2, c3, c4 = st.columns(4)
                        params["alpha"] = c1.number_input(
                            "alpha",
                            value=param_defaults.get("alpha", 2.0),
                            min_value=0.0001,
                            step=0.01,
                            format="%.2f",
                            key=f"{label}_a",
                        )
                        params["beta"] = c2.number_input(
                            "beta",
                            value=param_defaults.get("beta", 2.0),
                            min_value=0.0001,
                            step=0.01,
                            format="%.2f",
                            key=f"{label}_b",
                        )
                        params["min"] = c3.number_input("min (scale)", value=param_defaults.get("min", 0.0), key=f"{label}_bmin")
                        params["max"] = c4.number_input("max (scale)", value=param_defaults.get("max", 1.0), key=f"{label}_bmax")

                    elif dist_name == "Custom (P10/P50/P90)":
                        st.caption("Exceedance: **P10 = high (90th pct)**, **P90 = low (10th pct)**.")
                        fam = st.selectbox("Family", ["Normal", "Lognormal", "Beta"], key=f"{label}_cfam")
                        c1, c2, c3 = st.columns(3)
                        params["p90"] = c1.number_input("P90 (low)", value=param_defaults.get("p90", 1.0), key=f"{label}_c90")
                        params["p50"] = c2.number_input("P50 (median)", value=param_defaults.get("p50", 1.0), key=f"{label}_c50")
                        params["p10"] = c3.number_input("P10 (high)", value=param_defaults.get("p10", 1.0), key=f"{label}_c10")
                        params["family"] = fam
                        if fam == "Beta":
                            d1, d2 = st.columns(2)
                            params["min"] = d1.number_input("min (scale)", value=param_defaults.get("min", 0.0), key=f"{label}_cmin")
                            params["max"] = d2.number_input("max (scale)", value=param_defaults.get("max", 1.0), key=f"{label}_cmax")

                    elif dist_name == "Discrete":
                        st.caption("Define a set of discrete scenarios (values) with weights that sum to 100%.")
                        k = st.number_input("# of scenarios", min_value=1, max_value=20, value=int(param_defaults.get("k", 3)), step=1, key=f"{label}_disc_k")
                        import pandas as _pd
                        default_vals = param_defaults.get("values", [param_defaults.get("min", 1.0), param_defaults.get("mode", 1.0), param_defaults.get("max", 1.0)])
                        if len(default_vals) < int(k):
                            default_vals = (default_vals + [default_vals[-1] if default_vals else 1.0] * int(k))[:int(k)]
                        default_wts = param_defaults.get("weights", [100.0/float(k)]*int(k))
                        if len(default_wts) < int(k):
                            default_wts = (default_wts + [100.0/float(k)]*int(k))[:int(k)]
                        df_disc = _pd.DataFrame({"Value": default_vals[:int(k)], "Weight (%)": default_wts[:int(k)]})
                        edited_df = st.data_editor(
                            df_disc,
                            use_container_width=True,
                            num_rows="dynamic",
                            key=f"{label}_disc_table"
                        )
                        vals = _pd.to_numeric(edited_df["Value"], errors="coerce").fillna(0.0).astype(float).tolist()
                        wts = _pd.to_numeric(edited_df["Weight (%)"], errors="coerce").fillna(0.0).astype(float).tolist()
                        s = sum(wts)
                        probs = [wt/s if s>0 else 0.0 for wt in wts]
                        params["k"] = int(k)
                        params["values"] = vals
                        params["probs"] = probs

                except Exception:
                    pass

            is_valid = True
            fig_prev = None
            try:
                dist, ppf, pdf = make_distribution(dist_name.lower(), params)
                # Preview plot
                if dist_name == "Discrete":
                    xs = np.array(params.get("values", []), dtype=float)
                    ps = np.array(params.get("probs", []), dtype=float)
                    fig_prev = go.Figure()
                    fig_prev.add_trace(go.Bar(x=xs, y=ps, name=f"PMF {label}", marker_color=plot_color))
                    fig_prev.update_layout(title=f"{label} — Discrete Distribution",
                                           xaxis_title=label + (f" ({units_label})" if units_label else ""),
                                           yaxis_title="Probability")
                    st.plotly_chart(fig_prev, use_container_width=True)
                else:
                    x_min = float(dist.ppf(0.001))
                    x_max = float(dist.ppf(0.999))
                    xs = np.linspace(x_min, x_max, 400)
                    ys = pdf(xs)
                    fig_prev = go.Figure()
                    fig_prev.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"PDF {label}", line=dict(color=plot_color)))
                    fig_prev.update_layout(title=f"{label} — Distribution Preview",
                                           xaxis_title=label + (f" ({units_label})" if units_label else ""),
                                           yaxis_title="PDF")
                    st.plotly_chart(fig_prev, use_container_width=True)

                # Custom stats under the graph
                if dist_name == "Custom (P10/P50/P90)":
                    try: mu = float(np.squeeze(getattr(dist, "mean", lambda: np.nan)()))
                    except Exception: mu = np.nan
                    try: sd = float(np.squeeze(getattr(dist, "std", lambda: np.nan)()))
                    except Exception: sd = np.nan
                    try:
                        approx_min = float(np.squeeze(ppf(0.001)))
                        approx_max = float(np.squeeze(ppf(0.999)))
                    except Exception:
                        approx_min = np.nan; approx_max = np.nan
                    support = params.get("derived", {}).get("support", "—")
                    st.caption(f"**Estimated stats:** mean ≈ {mu:.6g}, std ≈ {sd:.6g}, approx min ≈ {approx_min:.6g}, approx max ≈ {approx_max:.6g}  |  support: {support}")
            except Exception:
                is_valid = False
                st.info("Select all parameters (and ensure they are consistent) to display the distribution preview.")

        return (dist if is_valid else None), (ppf if is_valid else None), dist_name, params, is_valid, fig_prev

    # ---------- Inputs depending on GRV mode ----------
    if grv_source == "Uncertain Area × Gross thickness":
        st.subheader("Input Distributions — Area × h")
        A_dist, A_ppf, A_name, A_params, A_ok, A_fig = dist_editor("Area (A)", 1, {"min": 5.0, "mode": 10.0, "max": 20.0}, units_label=u_area)
        h_dist, h_ppf, h_name, h_params, h_ok, h_fig = dist_editor("Gross Thickness (h)", 1, {"min": 10.0, "mode": 20.0, "max": 40.0}, units_label=u_thick)
        curve = None; curve_name = None; curve_ok = True
        use_grv_scale = False; grv_scale_ppf = None; grv_scale_params = {}
        GRV_ppf = None; GRV_ok = True

    elif grv_source == "Direct GRV distribution (m³)":
        st.subheader("GRV Mode — Direct GRV distribution")
        st.caption("Specify the distribution of **GRV in cubic meters**. This bypasses A×h and curve integration.")
        GRV_dist, GRV_ppf, GRV_name, GRV_params, GRV_ok, GRV_fig = dist_editor(
            "GRV (m³)", 4, {"median": 1e8, "sigma_ln": 0.5}, units_label="m³"
        )
        curve = None; curve_name = None; curve_ok = True
        use_grv_scale = False; grv_scale_ppf = None; grv_scale_params = {}
        A_ppf = h_ppf = None
        A_ok = h_ok = True

    else:
        st.subheader("Curve Mode — Upload a single Area–Depth curve")
        st.markdown(
            """
            **Expected file format**  
            Upload a CSV or Excel with **two columns**:
            - `depth` (in the selected depth units)
            - `area` (in the selected area units)
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            curve_file = st.file_uploader("Upload CSV/Excel with columns depth, area", type=["csv","xlsx","xls"], accept_multiple_files=False)
        with c2:
            st.caption("Depth & area units for the uploaded curve:")
            curve_depth_units = st.selectbox("Curve depth units", list(THICKNESS_UNITS.keys()), index=0, key="curve_depth_units")
            curve_area_units = st.selectbox("Curve area units", list(AREA_UNITS.keys()), index=list(AREA_UNITS.keys()).index(u_area), key="curve_area_units")

        curve = None; curve_name = None; curve_ok = False
        try:
            if curve_file is not None:
                dfc = _normalize_columns(_read_table(curve_file))
                dfc["depth_m"] = dfc["depth"].astype(float) * THICKNESS_UNITS[curve_depth_units]
                dfc["area_m2"]  = dfc["area"].astype(float) * AREA_UNITS[curve_area_units]
                dfc = dfc.dropna().sort_values("depth_m")
                curve = dfc[["depth_m","area_m2"]].to_numpy()
                curve_name = curve_file.name
                curve_ok = True
                curve_preview_fig = go.Figure()
                curve_preview_fig.add_trace(go.Scatter(x=curve[:,0], y=curve[:,1], mode="lines", name=f"{curve_name}"))
                curve_preview_fig.update_layout(title="Uploaded Area–Depth curve (SI)", xaxis_title="Depth (m)", yaxis_title="Area (m²)")
                st.plotly_chart(curve_preview_fig, use_container_width=True)
                base_grv_m3 = integrate_grv_m3(curve[:,0], curve[:,1], float(curve[:,0].min()), float(curve[:,0].max()))
                st.markdown(f"**Base GRV from curve (full depth range):** {base_grv_m3:,.0f} m³  &nbsp; | &nbsp; {base_grv_m3/1e6:,.3f} MMm³")
        except Exception as e:
            st.error(f"Failed to read curve: {e}")
            curve_ok = False

        A_ppf = h_ppf = None
        A_ok = h_ok = True
        GRV_ppf = None; GRV_ok = True
        grv_scale_ppf = None; grv_scale_params = {}; grv_scale_ok = True

    # ---------- Optional GRV scale factor (available for ALL GRV modes) ----------
    use_grv_scale = st.checkbox("Apply GRV scale factor (multiplier)?", value=False, key="use_grv_scale")
    if use_grv_scale:
        _, grv_scale_ppf, grv_scale_name, grv_scale_params, grv_scale_ok, grv_scale_fig = dist_editor(
            "GRV scale factor (multiplier)", 4, {"median": 1.0, "sigma_ln": 0.2}, units_label="×"
        )
    else:
        grv_scale_ppf = None; grv_scale_params = {}; grv_scale_ok = True; grv_scale_fig = None

    # --------- Other input dists ----------
    st.subheader("Other Input Distributions")
    NTG_dist, NTG_ppf, NTG_name, NTG_params, NTG_ok, NTG_fig = dist_editor("Net-to-Gross (NTG, fraction)", 2, {"min": 0.2, "mode": 0.5, "max": 0.8}, units_label="fraction")
    phi_dist, phi_ppf, phi_name, phi_params, phi_ok, phi_fig = dist_editor("Porosity (phi, fraction)", 2, {"min": 0.12, "mode": 0.18, "max": 0.26}, units_label="fraction")
    Sw_dist, Sw_ppf, Sw_name, Sw_params, Sw_ok, Sw_fig = dist_editor("Water Saturation (Sw, fraction)", 2, {"min": 0.15, "mode": 0.25, "max": 0.4}, units_label="fraction")

    if fluid == "Oil":
        Bo_dist, Bo_ppf, Bo_name, Bo_params, Bo_ok, Bo_fig = dist_editor("Oil FVF (Bo, rm³/stb)", 3, {"mean": 1.25, "sd": 0.1, "min": 1.05, "max": 1.5}, units_label="rm³/stb")
        Bg_ppf = None; Bg_ok = True; Bg_fig = None
    else:
        Bg_dist, Bg_ppf, Bg_name, Bg_params, Bg_ok, Bg_fig = dist_editor("Gas FVF (Bg, rm³/scf)", 3, {"mean": 0.004, "sd": 0.0004, "min": 0.003, "max": 0.0055}, units_label="rm³/scf")
        Bo_ppf = None; Bo_ok = True; Bo_fig = None

    # Optional RF
    include_rf = st.checkbox("Include Recovery Factor (compute recoverable volumes)", value=True, key="include_rf")
    if include_rf:
        RF_dist, RF_ppf, RF_name, RF_params, RF_ok, RF_fig = dist_editor("Recovery Factor (RF, fraction)", 4, {"median": 0.25, "sigma_ln": 0.3}, units_label="fraction")
    else:
        RF_ppf = None; RF_ok = True; RF_fig = None; RF_name = "—"; RF_params = {}

    # Optional multi-phase: Rs (oil) or CGR (gas)
    if fluid == "Oil":
        include_rs = st.checkbox("Include Solution GOR (Rs) to compute solution gas initially in place", value=False, key="include_rs")
        if include_rs:
            Rs_dist, Rs_ppf, Rs_name, Rs_params, Rs_ok, Rs_fig = dist_editor("Solution GOR (Rs, scf/stb)", 3, {"mean": 500.0, "sd": 100.0, "min": 0.0, "max": 3000.0}, units_label="scf/stb")
        else:
            Rs_ppf=None; Rs_ok=True; Rs_fig=None; Rs_name="—"; Rs_params={}
    else:
        include_cgr = st.checkbox("Include Condensate Yield (CGR) to compute condensate initially in place", value=False, key="include_cgr")
        if include_cgr:
            CGR_dist, CGR_ppf, CGR_name, CGR_params, CGR_ok, CGR_fig = dist_editor("Condensate Yield (CGR, STB/MMscf)", 3, {"mean": 50.0, "sd": 20.0, "min": 0.0, "max": 500.0}, units_label="STB/MMscf")
        else:
            CGR_ppf=None; CGR_ok=True; CGR_fig=None; CGR_name="—"; CGR_params={}

    # Collect figs safely for the report
    input_figs = {k:v for k,v in {
        "Area (A)": A_fig,
        "Gross Thickness (h)": h_fig,
        "Net-to-Gross (NTG, fraction)": NTG_fig,
        "Porosity (phi, fraction)": phi_fig,
        "Water Saturation (Sw, fraction)": Sw_fig,
        "Oil FVF (Bo, rm³/stb)": Bo_fig,
        "Gas FVF (Bg, rm³/scf)": Bg_fig,
        "Recovery Factor (RF, fraction)": RF_fig,
        "Solution GOR (Rs, scf/stb)": (Rs_fig if (fluid == "Oil" and 'include_rs' in locals() and include_rs) else None),
        "Condensate Yield (CGR, STB/MMscf)": (CGR_fig if (fluid == "Gas" and 'include_cgr' in locals() and include_cgr) else None),
        "GRV scale factor (multiplier)": (grv_scale_fig if use_grv_scale else None),
        "GRV (m³)": GRV_fig if ('GRV_fig' in locals() and GRV_fig is not None) else None,
    }.items() if v is not None}

    # =========================
    # Dependencies (add only the pairs you want)
    # =========================
    enable_deps = st.checkbox("Enable dependencies (correlation matrix)", value=True, key="enable_deps_inline")

    # Build the variable list for dependency editor and sampling
    if grv_source == "Uncertain Area × Gross thickness":
        var_names = ["A", "h", "NTG", "phi", "Sw", ("Bo" if fluid == "Oil" else "Bg")]
        inputs_ok = all([A_ok, h_ok, NTG_ok, phi_ok, Sw_ok, (Bo_ok if fluid=="Oil" else Bg_ok)])
    elif grv_source == "Direct GRV distribution (m³)":
        var_names = ["GRV_m3", "NTG", "phi", "Sw", ("Bo" if fluid == "Oil" else "Bg")]
        inputs_ok = all([GRV_ok, NTG_ok, phi_ok, Sw_ok, (Bo_ok if fluid=="Oil" else Bg_ok)])
    else:  # curve
        var_names = ["NTG", "phi", "Sw", ("Bo" if fluid == "Oil" else "Bg")]
        inputs_ok = all([curve_ok, NTG_ok, phi_ok, Sw_ok, (Bo_ok if fluid=="Oil" else Bg_ok)])

    if include_rf:
        var_names.append("RF")
        # RF_ok already set when include_rf True
        inputs_ok = inputs_ok and RF_ok
    if fluid == "Oil" and 'include_rs' in locals() and include_rs:
        var_names.append("Rs")
        inputs_ok = inputs_ok and Rs_ok
    if fluid == "Gas" and 'include_cgr' in locals() and include_cgr:
        var_names.append("CGR")
        inputs_ok = inputs_ok and CGR_ok
    if use_grv_scale:
        var_names.append("GRV_scale")
        inputs_ok = inputs_ok and grv_scale_ok

    k = len(var_names)

    # --- Dependency editor UI ---
    if enable_deps:
        st.caption("Add only the pairs you want to correlate. Unlisted pairs default to 0.00.")
        if "dep_rows" not in st.session_state:
            st.session_state.dep_rows = pd.DataFrame(columns=["Var i", "Var j", "rho"])

        with st.form("deps_form", clear_on_submit=False):
            dep_df = st.data_editor(
                st.session_state.dep_rows,
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True,
                key="dep_editor_rows",
                column_order=["Var i", "Var j", "rho"],
                column_config={
                    "Var i": st.column_config.SelectboxColumn(options=var_names),
                    "Var j": st.column_config.SelectboxColumn(options=var_names),
                    "rho": st.column_config.NumberColumn(min_value=-0.999, max_value=0.999, step=0.05, format="%.3f"),
                },
            )
            st.caption("Pick two variables and set rho; blank rho defaults to 0.000.")
            col_apply, col_reset = st.columns(2)
            with col_apply:
                submitted = st.form_submit_button("Apply dependencies")
            with col_reset:
                reset_clicked = st.form_submit_button("Reset dependencies", type="secondary")

        # If submitted, normalize and persist the edited rows
        if 'reset_clicked' in locals() and reset_clicked:
            st.session_state.dep_rows = pd.DataFrame(columns=["Var i", "Var j", "rho"])
            st.session_state._dep_pairs = []
            st.session_state._dep_vars = []
            st.session_state._has_deps = False
            st.success("Dependencies reset.")

        if 'dep_df' in locals() and submitted:
            df_norm = dep_df.copy()
            for col in ["Var i", "Var j"]:
                if col in df_norm.columns:
                    df_norm[col] = df_norm[col].astype(object).where(pd.notna(df_norm[col]), "")
            if "rho" in df_norm.columns:
                mask_fill0 = (
                    df_norm.get("Var i", "").astype(str).str.len() > 0
                ) & (
                    df_norm.get("Var j", "").astype(str).str.len() > 0
                ) & (
                    df_norm["rho"].isna()
                )
                df_norm.loc[mask_fill0, "rho"] = 0.0
            empty_mask = (
                (df_norm.get("Var i", "").astype(str).str.len() == 0)
                & (df_norm.get("Var j", "").astype(str).str.len() == 0)
                & (df_norm.get("rho").isna())
            ) if set(["Var i", "Var j", "rho"]).issubset(df_norm.columns) else pd.Series(False, index=df_norm.index)
            df_norm = df_norm[~empty_mask]
            # Reindex for neatness (hidden index, but keeps stable order)
            df_norm.index = np.arange(1, len(df_norm) + 1)
            st.session_state.dep_rows = df_norm.copy()
            st.success("Dependencies applied.", icon="✅")

        # Build correlation matrix from the persisted rows
        df_src = st.session_state.dep_rows.copy()
        conflicts = []
        seen = {}
        for _, r in df_src.iterrows():
            a = str(r.get("Var i", "")).strip()
            b = str(r.get("Var j", "")).strip()
            v = r.get("rho", 0.0)
            if not a or not b:
                continue
            if pd.isna(v):
                v = 0.0
            if a == b:
                continue
            if a not in var_names or b not in var_names:
                continue
            key_ab = tuple(sorted([a, b]))
            v = float(v)
            if key_ab in seen and abs(seen[key_ab] - v) > 1e-9:
                conflicts.append(f"{key_ab}: {seen[key_ab]:.3f} vs {v:.3f}")
            else:
                seen[key_ab] = v
        if conflicts:
            st.error("Conflicting entries for the same pair: " + "; ".join(conflicts))

        C = np.eye(k)
        for (a, b), v in seen.items():
            i, j = var_names.index(a), var_names.index(b)
            C[i, j] = C[j, i] = max(-0.999, min(0.999, float(v)))
        np.fill_diagonal(C, 1.0)
        C_pd = nearest_pd(C)
        try:
            st.session_state._dep_pairs = list(seen.keys())
            st.session_state._dep_vars = sorted({x for pair in seen.keys() for x in pair})
            st.session_state._has_deps = len(seen) > 0
        except Exception:
            st.session_state._dep_pairs = []
            st.session_state._dep_vars = []
            st.session_state._has_deps = False
    else:
        C_pd = np.eye(k)
        # No dependency editor or disabled: clear stored selections
        st.session_state._dep_pairs = []
        st.session_state._dep_vars = []
        st.session_state._has_deps = False

    st.divider()

    # =========================
    # RUN — Monte Carlo
    # =========================
    run_btn = st.button("Run Monte Carlo", type="primary", use_container_width=True, disabled=(not inputs_ok))
    if not inputs_ok:
        st.info("Fill in/validate all required input distributions first.")

    if run_btn and inputs_ok:
        # Build the list of PPFs according to var_names order
        name_to_ppf = {}
        if grv_source == "Uncertain Area × Gross thickness":
            name_to_ppf.update({"A": A_ppf, "h": h_ppf})
        elif grv_source == "Direct GRV distribution (m³)":
            name_to_ppf.update({"GRV_m3": GRV_ppf})
        # Common ones
        name_to_ppf.update({
            "NTG": NTG_ppf,
            "phi": phi_ppf,
            "Sw": Sw_ppf,
            ("Bo" if fluid == "Oil" else "Bg"): (Bo_ppf if fluid == "Oil" else Bg_ppf),
        })
        if include_rf:
            name_to_ppf["RF"] = RF_ppf
        if fluid == "Oil" and 'include_rs' in locals() and include_rs:
            name_to_ppf["Rs"] = Rs_ppf
        if fluid == "Gas" and 'include_cgr' in locals() and include_cgr:
            name_to_ppf["CGR"] = CGR_ppf
        if use_grv_scale:
            name_to_ppf["GRV_scale"] = grv_scale_ppf

        ppfs = [name_to_ppf[nm] for nm in var_names]

        # LHS uniforms + Gaussian copula transform
        U = latin_hypercube(int(n), k, rng)
        X = gaussian_copula_sample(int(n), C_pd, ppfs, rng, lhs_uniforms=U)

        samples = pd.DataFrame(X, columns=var_names)

        # Unit conversions & rename for compute_volumes
        if "A" in samples.columns:
            samples["A_m2"] = samples["A"].astype(float) * AREA_UNITS[u_area]
        if "h" in samples.columns:
            samples["h_m"] = samples["h"].astype(float) * THICKNESS_UNITS[u_thick]
        if "Bo" in samples.columns:
            samples["Bo_rm3_per_stb"] = samples["Bo"].astype(float)
        if "Bg" in samples.columns:
            samples["Bg_rm3_per_scf"] = samples["Bg"].astype(float)

        # Geometry handling per mode
        if grv_source == "Uncertain Area × Gross thickness":
            if use_grv_scale and "GRV_scale" in samples.columns:
                # Collapse geometry into GRV for compute_volumes
                samples["GRV_m3"] = (samples["A_m2"] * samples["h_m"]) * samples["GRV_scale"]
            # else: leave A_m2 and h_m for compute_volumes path

        elif grv_source == "Direct GRV distribution (m³)":
            # Already sampled GRV_m3; multiply by scale if requested
            if use_grv_scale and "GRV_scale" in samples.columns:
                samples["GRV_m3"] = samples["GRV_m3"] * samples["GRV_scale"]

        else:
            # Curve mode — base GRV is deterministic from full depth range
            try:
                base_grv_m3 = integrate_grv_m3(curve[:,0], curve[:,1], float(curve[:,0].min()), float(curve[:,0].max()))
            except Exception:
                base_grv_m3 = 0.0
            if use_grv_scale and "GRV_scale" in samples.columns:
                samples["GRV_m3"] = base_grv_m3 * samples["GRV_scale"]
            else:
                samples["GRV_m3"] = float(base_grv_m3)

        # Clip fractions
        for col in ["NTG", "phi", "Sw"] + (["RF"] if include_rf else []):
            if col in samples.columns:
                samples[col] = samples[col].clip(0.0, 1.0)

        # Compute volumes (includes SGIIP/CIP if Rs/CGR present)
        vols = compute_volumes(samples, fluid, include_rf=include_rf)
        out = pd.concat([samples, vols], axis=1)

        st.success("Simulation complete.")
        # =========
        # Grid of charts for all relevant outputs (no selector)
        # Also collect figures for the HTML report
        # =========
        report_figs = []

        charts = []  # (title, series, decimals)
        if fluid == "Oil":
            # Base in‑place
            if out_unit in {"m³","Mm³","MMm³"}:
                charts.append((f"STOIIP ({out_unit})", out["STOIIP_m3"] * (OIL_UNITS[out_unit]/OIL_UNITS["m³"]), int(decimals)))
            else:
                charts.append((f"STOIIP ({out_unit})", out["STOIIP_stb"] * (OIL_UNITS[out_unit]/OIL_UNITS["stb"]), int(decimals)))
            # Reserves
            if include_rf and ("Reserves_m3" in out.columns or "Reserves_stb" in out.columns):
                if out_unit in {"m³","Mm³","MMm³"} and "Reserves_m3" in out.columns:
                    charts.append((f"Reserves ({out_unit})", out["Reserves_m3"] * (OIL_UNITS[out_unit]/OIL_UNITS["m³"]), int(decimals)))
                elif "Reserves_stb" in out.columns:
                    charts.append((f"Reserves ({out_unit})", out["Reserves_stb"] * (OIL_UNITS[out_unit]/OIL_UNITS["stb"]), int(decimals)))
            # Solution gas (if Rs)
            if "SGIIP_scf" in out.columns and st.session_state.get("include_rs", False):
                sg_unit_local = st.session_state.get("sg_unit", "MMscf")
                sg_decs_local = int(st.session_state.get("sg_decimals", 2))
                charts.append((f"Solution gas ({sg_unit_local})", out["SGIIP_scf"] / GAS_UNITS["scf"] * GAS_UNITS[sg_unit_local], sg_decs_local))
        else:
            # Gas base
            if out_unit in {"m³","Mm³","MMm³"}:
                charts.append((f"GIIP ({out_unit})", out["GIIP_m3"] * (GAS_UNITS[out_unit]/GAS_UNITS["m³"]), int(decimals)))
            else:
                charts.append((f"GIIP ({out_unit})", out["GIIP_scf"] * (GAS_UNITS[out_unit]/GAS_UNITS["scf"]), int(decimals)))
            # Recoverable
            if include_rf and ("Recoverable_m3" in out.columns or "Recoverable_scf" in out.columns):
                if out_unit in {"m³","Mm³","MMm³"} and "Recoverable_m3" in out.columns:
                    charts.append((f"Recoverable gas ({out_unit})", out["Recoverable_m3"] * (GAS_UNITS[out_unit]/GAS_UNITS["m³"]), int(decimals)))
                elif "Recoverable_scf" in out.columns:
                    charts.append((f"Recoverable gas ({out_unit})", out["Recoverable_scf"] * (GAS_UNITS[out_unit]/GAS_UNITS["scf"]), int(decimals)))
            # Condensate (if CGR)
            if "Condensate_stb" in out.columns and st.session_state.get("include_cgr", False):
                cnd_unit_local = st.session_state.get("cnd_unit", "Mstb")
                cnd_decs_local = int(st.session_state.get("cnd_decimals", 2))
                charts.append((f"Condensate ({cnd_unit_local})", out["Condensate_stb"] / OIL_UNITS["stb"] * OIL_UNITS[cnd_unit_local], cnd_decs_local))

        # Render in 2‑column rows: each item shows Histogram and CDF side by side
        for title, series, decs in charts:
            st.markdown(f"### {title}")
            qP90 = float(np.quantile(series, 0.10))
            qP50 = float(np.quantile(series, 0.50))
            qP10 = float(np.quantile(series, 0.90))

            c1, c2 = st.columns(2)
            with c1:
                hist_fig = px.histogram(x=series, nbins=60, title=f"Histogram — {title}")
                hist_fig.update_traces(marker_color=plot_color, hovertemplate="Value=%{x:,.3f}<br>Count=%{y}")
                hist_fig.add_vline(qP90, line_dash="dot", line_color=annot_color, annotation_text="P90", annotation_position="top left")
                hist_fig.add_vline(qP50, line_dash="dash", line_color=annot_color, annotation_text="P50", annotation_position="top left")
                hist_fig.add_vline(qP10, line_dash="dot", line_color=annot_color, annotation_text="P10", annotation_position="top left")
                hist_fig.update_layout(xaxis_title=title, yaxis_title="Count", height=320)
                st.plotly_chart(hist_fig, use_container_width=True)
                report_figs.append(hist_fig)
            with c2:
                xs = np.sort(np.asarray(series)); ys = np.linspace(1.0/len(xs), 1.0, len(xs))
                cdf_fig = go.Figure()
                cdf_fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="CDF"))
                cdf_fig.add_vline(qP90, line_dash="dot", line_color=annot_color, annotation_text="P90", annotation_position="top left")
                cdf_fig.add_vline(qP50, line_dash="dash", line_color=annot_color, annotation_text="P50", annotation_position="top left")
                cdf_fig.add_vline(qP10, line_dash="dot", line_color=annot_color, annotation_text="P10", annotation_position="top left")
                cdf_fig.update_layout(title=f"CDF — {title}", xaxis_title=title, yaxis_title="P(X ≤ x)", height=320)
                st.plotly_chart(cdf_fig, use_container_width=True)
                report_figs.append(cdf_fig)

        # =========
        # Correlation heatmap (inputs vs base in-place) — Viridis, wider
        # =========
        # Only include variables that impact base in‑place (STOIIP/GIIP)
        # Oil: geometry, NTG, phi, Sw, Bo
        # Gas: geometry, NTG, phi, Sw, Bg
        corr_inputs = []
        if "GRV_m3" in out.columns:
            corr_inputs.append("GRV_m3")
        else:
            if "A" in var_names:
                corr_inputs.append("A")
            if "h" in var_names:
                corr_inputs.append("h")
        for nm in ["NTG", "phi", "Sw"]:
            if nm in var_names and nm not in corr_inputs:
                corr_inputs.append(nm)
        if fluid == "Oil":
            if "Bo" in var_names and "Bo" not in corr_inputs:
                corr_inputs.append("Bo")
        else:
            if "Bg" in var_names and "Bg" not in corr_inputs:
                corr_inputs.append("Bg")

        alias = {
            "A": f"A ({u_area})",
            "h": f"h ({u_thick})",
            "GRV_m3": f"GRV ({u_area}×{u_thick})" if ("A" in var_names or "h" in var_names) else "GRV (m³)",
            "NTG": "NTG (frac)",
            "phi": "Phi (frac)",
            "Sw": "Sw (frac)",
            "Bo": "Bo (rm³/stb)",
            "Bg": "Bg (rm³/scf)",
            "RF": "RF (frac)",
            "Rs": "Rs (scf/stb)",
            "CGR": "CGR (STB/MMscf)",
        }

        # Choose base in‑place as target for comparability
        if fluid == "Oil":
            target_series = out["STOIIP_m3"] if out_unit in {"m³","Mm³","MMm³"} else out["STOIIP_stb"]
            target_label = f"STOIIP ({out_unit})"
        else:
            target_series = out["GIIP_m3"] if out_unit in {"m³","Mm³","MMm³"} else out["GIIP_scf"]
            target_label = f"GIIP ({out_unit})"

        corr_df = out[[c for c in corr_inputs if c in out.columns]].copy()
        corr_df[target_label] = target_series.values
        corr_mat = corr_df.corr(method="spearman").round(int(st.session_state.get("heatmap_decimals", 1)))
        corr_mat.index = [alias.get(c, c) for c in corr_mat.index]
        corr_mat.columns = [alias.get(c, c) if c in corr_inputs else target_label for c in corr_mat.columns]

        # Decide what to display based on user-selected dependencies
        has_deps = bool(st.session_state.get("_has_deps", False))
        dep_vars = list(st.session_state.get("_dep_vars", []))

        if not has_deps:
            # Show only Inputs vs Target as a single column heatmap
            # Take the target column and drop the target row
            display_df = corr_mat[[target_label]].copy()
            if target_label in display_df.index:
                display_df = display_df.drop(index=target_label)
            heat_title = "Correlation heatmap (inputs vs base in‑place) — inputs vs target only"
        else:
            # When dependencies are defined, show the full correlation matrix
            display_df = corr_mat
            heat_title = "Correlation heatmap (full matrix)"

        heat = px.imshow(
            display_df,
            text_auto=True,
            color_continuous_scale="RdBu",
            range_color=[-1.0, 1.0],
            color_continuous_midpoint=0.0,
            aspect="auto",
            title=heat_title,
        )
        # Increase readability of in-cell correlation text with adaptive sizing
        try:
            n_rows, n_cols = display_df.shape
            max_dim = max(int(n_rows), int(n_cols))
            if max_dim <= 5:
                text_size = 16
            elif max_dim <= 8:
                text_size = 14
            elif max_dim <= 12:
                text_size = 12
            else:
                text_size = 10
            heat.update_traces(textfont_size=text_size)
        except Exception:
            pass
        heat.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=600)
        st.plotly_chart(heat, use_container_width=True)
        report_figs.append(heat)

        # =========
        # Tornado — ΔP50 % (base in-place), exclude RF and Rs/CGR, sorted by impact w/ rich hover
        # =========
        # Build base series scaled to the chosen unit for the in-place target
        if fluid == "Oil":
            if out_unit in {"m³","Mm³","MMm³"}:
                base_series = out["STOIIP_m3"] * (OIL_UNITS[out_unit] / OIL_UNITS["m³"])  # m³ → chosen
            else:
                base_series = out["STOIIP_stb"] * (OIL_UNITS[out_unit] / OIL_UNITS["stb"])  # stb → chosen
        else:
            if out_unit in {"m³","Mm³","MMm³"}:
                base_series = out["GIIP_m3"] * (GAS_UNITS[out_unit] / GAS_UNITS["m³"])  # m³ → chosen
            else:
                base_series = out["GIIP_scf"] * (GAS_UNITS[out_unit] / GAS_UNITS["scf"])  # scf → chosen
        base_p50 = float(np.quantile(base_series, 0.50))

        st.subheader("Sensitivity — Tornado (ΔP50 %)")

        def recompute_p50_with_overrides(df_base, var, val):
            df = df_base.copy()
            # Set variable and its SI companion columns when applicable
            if var == "A":
                df["A"] = val
                df["A_m2"] = val * AREA_UNITS[u_area]
            elif var == "h":
                df["h"] = val
                df["h_m"] = val * THICKNESS_UNITS[u_thick]
            elif var in ("Bo", "Bg"):
                df[var] = val
                if var == "Bo":
                    df["Bo_rm3_per_stb"] = val
                else:
                    df["Bg_rm3_per_scf"] = val
            elif var == "GRV_m3":
                df["GRV_m3"] = val
            else:
                df[var] = val

            # If using GRV scaling in Area×h mode, keep GRV_m3 consistent when A or h change
            if (
                grv_source == "Uncertain Area × Gross thickness" and
                use_grv_scale and
                var in ("A", "h") and
                ("GRV_m3" in df.columns) and ("GRV_scale" in df.columns) and
                ("A_m2" in df.columns) and ("h_m" in df.columns)
            ):
                df["GRV_m3"] = df["A_m2"] * df["h_m"] * df["GRV_scale"]

            # keep physical bounds on fractions
            for c in ["NTG", "phi", "Sw", "RF"]:
                if c in df.columns:
                    df[c] = df[c].clip(0, 1)

            vols_tmp = compute_volumes(df, fluid, include_rf=False)  # base in-place only
            if fluid == "Oil":
                if out_unit in {"m³","Mm³","MMm³"}:
                    x_scaled_tmp = vols_tmp["STOIIP_m3"] * (OIL_UNITS[out_unit] / OIL_UNITS["m³"])
                else:
                    x_scaled_tmp = vols_tmp["STOIIP_stb"] * (OIL_UNITS[out_unit] / OIL_UNITS["stb"])
            else:
                if out_unit in {"m³","Mm³","MMm³"}:
                    x_scaled_tmp = vols_tmp["GIIP_m3"] * (GAS_UNITS[out_unit] / GAS_UNITS["m³"])
                else:
                    x_scaled_tmp = vols_tmp["GIIP_scf"] * (GAS_UNITS[out_unit] / GAS_UNITS["scf"])
            return float(np.quantile(x_scaled_tmp, 0.50))

        # Variables to test: geometry + NTG + phi + Sw + Bo/Bg (exclude RF, Rs, CGR)
        if use_grv_scale:
            base_list = ["GRV_m3", "NTG", "phi", "Sw", "Bo", "Bg"]
        else:
            base_list = ["A", "h", "GRV_m3", "NTG", "phi", "Sw", "Bo", "Bg"]
        sens_vars = [v for v in base_list if v in samples.columns]

        grv_unit_label = f"{u_area}×{u_thick}" if use_grv_scale else "m³"
        _alias = {"GRV_m3": f"GRV ({grv_unit_label})"}

        # Compute low/high deltas, plus hover fields
        tor_records = []
        for v in sens_vars:
            v_low = float(np.quantile(samples[v], 0.10))
            v_high = float(np.quantile(samples[v], 0.90))
            p50_low = recompute_p50_with_overrides(samples, v, v_low)
            p50_high = recompute_p50_with_overrides(samples, v, v_high)
            d_low_pct = (p50_low / base_p50 - 1.0) * 100.0
            d_high_pct = (p50_high / base_p50 - 1.0) * 100.0
            tor_records.append({
                "Variable": _alias.get(v, v),
                "Low %": d_low_pct,
                "High %": d_high_pct,
                "P10 value": v_low,
                "P90 value": v_high,
                "P50 at low": p50_low,
                "P50 at high": p50_high,
                "Impact": max(abs(d_low_pct), abs(d_high_pct)),
            })

        df_tor = pd.DataFrame(tor_records)
        # Sort by impact descending and fix y-category order so largest appears on top
        df_tor = df_tor.sort_values("Impact", ascending=False).reset_index(drop=True)

        # Build customdata for rich hover per trace
        low_custom  = np.column_stack([df_tor["P10 value"].values, df_tor["P50 at low"].values,  np.full(len(df_tor), base_p50)])
        high_custom = np.column_stack([df_tor["P90 value"].values, df_tor["P50 at high"].values, np.full(len(df_tor), base_p50)])

        fig_tornado = go.Figure()
        fig_tornado.add_trace(go.Bar(
            y=df_tor["Variable"], x=df_tor["Low %"], orientation='h', name='Low (P10 value)', marker_color="#d62728",
            customdata=low_custom,
            hovertemplate="ΔP50: %{x:.2f}%<br>P10 value: %{customdata[0]:,.6g}<br>P50 at low: %{customdata[1]:,.6g}<br>Base P50: %{customdata[2]:,.6g}<extra></extra>"
        ))
        fig_tornado.add_trace(go.Bar(
            y=df_tor["Variable"], x=df_tor["High %"], orientation='h', name='High (P90 value)', marker_color=plot_color,
            customdata=high_custom,
            hovertemplate="ΔP50: %{x:.2f}%<br>P90 value: %{customdata[0]:,.6g}<br>P50 at high: %{customdata[1]:,.6g}<br>Base P50: %{customdata[2]:,.6g}<extra></extra>"
        ))

        fig_tornado.update_layout(
            barmode='overlay',
            title="Tornado: % change in P50 of in-place when each input is set to its P10/P90",
            xaxis_title="ΔP50 (%) relative to base",
            yaxis_title="Input variable",
            yaxis=dict(categoryorder='array', categoryarray=df_tor["Variable"][::-1].tolist())
        )
        fig_tornado.add_vline(x=0.0, line_color="#888", line_dash="dash")
        st.plotly_chart(fig_tornado, use_container_width=True)
        report_figs.append(fig_tornado)

        # =========
        # Results table (all computed outputs) + CSV
        # =========
        def scale_series(series, system):
            """Scale an in-place or recoverable series to the selected output unit.
            For Oil: input series is passed in either m³ or stb depending on caller.
            For Gas: input series is passed in either m³ or scf depending on caller.
            """
            metric_keys = {"m³", "Mm³", "MMm³"}
            if system == "Oil":
                if out_unit in metric_keys:
                    # Series expected in m³
                    return series * (OIL_UNITS[out_unit] / OIL_UNITS["m³"])
                else:
                    # Series expected in stb
                    return series * (OIL_UNITS[out_unit] / OIL_UNITS["stb"])
            else:  # Gas
                if out_unit in metric_keys:
                    # Series expected in m³
                    return series * (GAS_UNITS[out_unit] / GAS_UNITS["m³"])
                else:
                    # Series expected in scf
                    return series * (GAS_UNITS[out_unit] / GAS_UNITS["scf"])
        def four_stats(arr):
            return pd.Series({
                "P10 (high)": np.quantile(arr, 0.90),
                "P50": np.quantile(arr, 0.50),
                "P90 (low)": np.quantile(arr, 0.10),
                "Mean": np.mean(arr),
            })

        results_cols = {}
        # Oil group
        if fluid == "Oil":
            if out_unit in {"m³","Mm³","MMm³"}:
                results_cols[f"STOIIP ({out_unit})"] = scale_series(out["STOIIP_m3"], "Oil")
                if include_rf and "Reserves_m3" in out.columns:
                    results_cols[f"Reserves ({out_unit})"] = scale_series(out["Reserves_m3"], "Oil")
            else:
                results_cols[f"STOIIP ({out_unit})"] = scale_series(out["STOIIP_stb"], "Oil")
                if include_rf and "Reserves_stb" in out.columns:
                    results_cols[f"Reserves ({out_unit})"] = scale_series(out["Reserves_stb"], "Oil")
            # Solution gas (unit-converted, remove raw)
            if "SGIIP_scf" in out.columns and st.session_state.get("include_rs", False):
                sg_unit_local = st.session_state.get("sg_unit", "MMscf")
                # Convert from scf base to chosen unit (scf or metric)
                if sg_unit_local in {"m³","Mm³","MMm³"}:
                    results_cols[f"Solution gas ({sg_unit_local})"] = out["SGIIP_scf"] / GAS_UNITS["scf"] * GAS_UNITS[sg_unit_local]
                else:
                    results_cols[f"Solution gas ({sg_unit_local})"] = out["SGIIP_scf"] / GAS_UNITS["scf"] * GAS_UNITS[sg_unit_local]
        else:
            if out_unit in {"m³","Mm³","MMm³"}:
                results_cols[f"GIIP ({out_unit})"] = scale_series(out["GIIP_m3"], "Gas")
                if include_rf and "Recoverable_m3" in out.columns:
                    results_cols[f"Recoverable gas ({out_unit})"] = scale_series(out["Recoverable_m3"], "Gas")
            else:
                results_cols[f"GIIP ({out_unit})"] = scale_series(out["GIIP_scf"], "Gas")
                if include_rf and "Recoverable_scf" in out.columns:
                    results_cols[f"Recoverable gas ({out_unit})"] = scale_series(out["Recoverable_scf"], "Gas")
            # Condensate (unit-converted, remove raw)
            if "Condensate_stb" in out.columns and st.session_state.get("include_cgr", False):
                cnd_unit_local = st.session_state.get("cnd_unit", "Mstb")
                if cnd_unit_local in {"m³","Mm³","MMm³"}:
                    results_cols[f"Condensate ({cnd_unit_local})"] = out["Condensate_stb"] / OIL_UNITS["stb"] * OIL_UNITS[cnd_unit_local]
                else:
                    results_cols[f"Condensate ({cnd_unit_local})"] = out["Condensate_stb"] / OIL_UNITS["stb"] * OIL_UNITS[cnd_unit_local]

        st.subheader("Results (exceedance: P90 low, P10 high)")
        res_df = pd.concat({k: four_stats(v) for k, v in results_cols.items()}, axis=1)
        st.table(res_df.style.format("{:,.%df}" % int(decimals)))

        # CSV + HTML report — combined ZIP download to prevent rerun resets
        out_scaled = out.copy()
        for k, v in results_cols.items():
            out_scaled[k] = v.values if isinstance(v, pd.Series) else v
        csv_bytes = out_scaled.to_csv(index=False).encode("utf-8")

        html_sections = [
            f"<h1>{APP_NAME} — {APP_VER}</h1>",
            f"<p>Run date: {datetime.utcnow().isoformat()}Z</p>",
            f"<h2>Summary (exceedance)</h2>",
            res_df.to_html(border=0, classes='table table-striped', float_format=("{:,.%df}" % int(decimals)).format)
        ]
        for i, fig in enumerate(report_figs):
            html_sections.append(pio.to_html(fig, include_plotlyjs=(i==0), full_html=False))
        html_report = "\n".join(html_sections).encode("utf-8")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("mc_samples.csv", csv_bytes)
            zf.writestr("volumetrics_report.html", html_report)
        zip_buf.seek(0)

        st.download_button(
            "Download samples & report (ZIP)",
            data=zip_buf.getvalue(),
            file_name="samples_and_report.zip",
            mime="application/zip"
        )

with help_tab:
    st.header("User Guide")
    st.markdown(
        r"""
### What this app does
Monte‑Carlo simulator for subsurface volumetrics. It supports three GRV workflows, optional **GRV scaling**, dependencies between inputs, optional **RF**, and multi‑phase add‑ons (**Rs → Solution gas** in oil systems; **CGR → Condensate** in gas systems). Charts (Histogram, CDF, Heatmap, Tornado) and the results table react to your selections.

---
### Quick start
1) **Pick fluid system** in the sidebar (Oil or Gas) and set **units** and **iterations**.
2) **Choose GRV source** under *GRV Source*:
   - **Uncertain Area × Gross thickness** — provide distributions for **A** and **h**.
   - **Upload Area–Depth curve** — upload a CSV/XLSX with two columns: `depth` and `area`. Use the unit selectors next to the uploader. The app integrates area over depth to compute GRV.
   - **Direct GRV distribution (m³)** — provide a distribution for GRV in m³.
3) (Optional) **Apply GRV scale factor** — a multiplicative “fudge factor” on GRV. If scaling is used, *sensitivity charts treat geometry as a single GRV input* (see Notes below).
4) Define **Other inputs**: NTG, phi, Sw, and **Bo** (Oil) or **Bg** (Gas). Use any supported distribution (Uniform, Triangular, PERT, Normal (optional bounds), Lognormal (optional bounds), Beta, Custom P10/P50/P90, Discrete).
5) (Optional) **Recovery Factor (RF)** — enable to compute reserves/recoverable volumes.
6) (Optional multi‑phase)**:**
   - **Oil**: enable **Rs (scf/stb)** to compute **SGIIP**.
   - **Gas**: enable **CGR (STB/MMscf)** to compute **Condensate in place**.
7) (Optional) **Dependencies** — toggle *Enable dependencies* and add only the pairs you want. Unlisted pairs default to 0.00.
8) Click **Run Monte Carlo**.
9) Choose the **Target output** (drop‑down) to drive charts and the tornado baseline (P50 of the selected output).

---
### Inputs & units
- **Area units:** m², km², acres.  
- **Thickness units:** m, ft.  
- **Oil outputs units:** m³ / Mm³ / MMm³ or stb / Mstb / MMstb (conversion from 1 m³ = 6.2898 stb).  
- **Gas outputs units:** m³ / Mm³ / MMm³ or scf / MMscf / Bscf / Tscf (conversion from 1 m³ = 35.3147 scf).  
- **Curve file format:** CSV or Excel with columns named like `depth`, `area` (case‑insensitive). Select curve units next to the uploader. The app displays the uploaded curve and reports the integrated GRV over its full depth range.

---
### Dependency editor (correlations)
- Only pairs you add are applied; others remain **0.00** by default.  
- Each row: **Var i**, **Var j**, **rho** (−0.999…+0.999).  
- The editor checks for **conflicting duplicates** of the same pair and warns.  
- The correlation matrix is adjusted to the nearest positive‑semidefinite matrix before sampling.  
- Your rows persist via session state while the app remains open.

---
### Calculations
- **HC pore volume (reservoir m³)** = GRV × NTG × phi × (1 − Sw).  
- **Oil system**:  
  - **STOIIP** (std conditions) = HCPV / Bo.  
  - **SGIIP** (scf) = STOIIP(stb) × Rs.  
  - **Reserves** (if RF enabled) = STOIIP × RF.  
- **Gas system**:  
  - **GIIP** (std conditions) = HCPV / Bg.  
  - **Condensate in place** (STB) = GIIP(scf)/1e6 × CGR.  
  - **Recoverable gas** (if RF enabled) = GIIP × RF.

---
### Charts & tables
- **Target selector** drives:
  - **Histogram** and **CDF** (annotated with P10/P50/P90 — *exceedance convention*: P90 low, P10 high).  
  - **Correlation heatmap**: Spearman correlation of inputs vs the selected output.  
  - **Tornado**: Shows ΔP50 when each input is set to its P10/P90.  
- **GRV scaling behavior in charts**  
  - If **GRV scaling is OFF** and you used **A × h**, the heatmap and tornado show **A** and **h** separately.  
  - If **GRV scaling is ON** (any GRV method), geometry sensitivity is aggregated under **GRV**; **A** and **h** bars are hidden to avoid double‑counting.  
- **Results table** reports P10/P50/P90/Mean for all computed outputs relevant to your selections.  
- **CSV download** includes all sampled inputs plus computed outputs; additional scaled columns matching the table are appended for convenience.

---
### Tips
- Clip fractions (NTG, phi, Sw, RF) are automatically bounded to [0, 1].  
- Use **Custom (P10/P50/P90)** when eliciting inputs in exceedance terms (P10 high, P90 low).  
- For highly skewed parameters, prefer **Lognormal** (optionally truncated with bounds).

---
### Version & contact
- **App version:** {1.1.5}
- **Contact the project maintainers via email for support.**
        """
    )
