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
4) Define **Other inputs**: NTG, phi, Sw, and **Bo** (Oil) or **Bg** (Gas). Use any supported distribution (Uniform, Triangular, PERT, Normal/Truncated, Lognormal/Truncated, Beta, Custom P10/P50/P90, Discrete).
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
- For highly skewed parameters, prefer **Lognormal** or **Truncated Lognormal**.

---
### Version & contact
- **App version:** {1.1}
- **Contact the project maintainers via email for support.**
        """