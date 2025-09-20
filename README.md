## User Guide

### Overview
This application performs Monte Carlo volumetric assessments for subsurface reservoirs. It brings together three GRV workflows, rich input distributions, optional recovery/secondary products, and dependency-aware sampling so you can explore uncertainty and sensitivities in one place. Everything you configure in the Simulator tab updates the visual analytics and the downloadable report.

---
### How It Works
1. **Configure the study** in the sidebar: set iterations, random seed, fluid system, and unit conventions.
2. **Select a GRV workflow** (area × thickness, uploaded area–depth curve, or a direct GRV distribution). Optionally apply a GRV scale factor to all modes.
3. **Describe subsurface properties** by picking distributions for NTG, porosity, water saturation, and fluid formation volume factors. Add recovery factor or multi-phase add-ons (solution gas `Rs`, condensate yield `CGR`) as needed.
4. **Optionally define dependencies** between inputs. A Gaussian copula embeds your declared correlations into the sampling space.
5. **Run the Monte Carlo**. Latin Hypercube Sampling (LHS) generates stratified draws across the high-dimensional joint distribution, preserving the correlation structure from the copula.
6. **Review the outputs**: histograms, CDFs, heat maps, tornado sensitivities, and summary tables adapt to your selected output unit. Export samples and an HTML report when you are satisfied.

---
### What You Can Do
- Test scenarios by mixing and matching GRV sources with or without scale multipliers.
- Quantify uncertainty across STOIIP / GIIP, reserves, solution gas, or condensate volumes.
- Explore which inputs move the P50 most via the tornado chart.
- Share findings through ready-to-download CSV samples and a self-contained HTML report.

---
### Distribution Library (with references)
- **Uniform (random)** — equal likelihood between `min` and `max`. Useful when only bounds are known. [Read more](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
- **Triangular** — linear up/down between `min`, `mode`, and `max`; captures simple expert estimates. [Read more](https://en.wikipedia.org/wiki/Triangular_distribution)
- **PERT** — smooths a triangular estimate into a beta curve controlled by `lambda`. Ideal for elicited min/mode/max inputs. [Read more](https://en.wikipedia.org/wiki/PERT_distribution)
- **Normal** — Gaussian with optional truncation; use when symmetric uncertainty around a mean is appropriate. [Read more](https://en.wikipedia.org/wiki/Normal_distribution)
- **Lognormal** — positive skewed distribution defined by median and log-space sigma; optional bounds avoid extreme tails. [Read more](https://en.wikipedia.org/wiki/Log-normal_distribution)
- **Beta** — bounded on `[min, max]`, flexible for fractions such as NTG, porosity, or Sw. [Read more](https://en.wikipedia.org/wiki/Beta_distribution)
- **Custom (P10/P50/P90)** — back-calculates parameters for Normal, Lognormal, or Beta families from exceedance-style quantiles (P90 low, P10 high). [Read more](https://en.wikipedia.org/wiki/Quantile)
- **Discrete** — finite scenarios with explicit weights; use for structured cases or deterministic options. [Read more](https://en.wikipedia.org/wiki/Probability_mass_function)

Each expander in the Simulator tab previews the probability density (or PMF) so you can validate shapes before running.

---
### Sampling & Dependencies
- **Latin Hypercube Sampling (LHS)** stratifies each marginal distribution into equal-probability bins and samples once per bin, dramatically improving coverage relative to naïve Monte Carlo for the same iteration count. [Read more](https://en.wikipedia.org/wiki/Latin_hypercube_sampling)
- **Gaussian Copula** — user-defined correlations are assembled into a covariance matrix and adjusted to the nearest positive semi-definite matrix. Multivariate normal samples are drawn, correlated via a Cholesky factor, transformed to uniform space, and finally mapped through each variable’s inverse CDF (PPF). This preserves your rank correlations while honoring the individual distribution shapes. [Read more](https://en.wikipedia.org/wiki/Copula_(probability_theory))
- **Dependency editor** — add only the pairs you need. Unspecified pairs default to zero correlation. Conflicts are flagged, and session state remembers your entries until you reset them.

---
### Visual Guide: Sampling & Dependencies

#### Latin Hypercube Sampling vs Random Sampling
The charts below contrast 200 random draws from the unit square with Latin Hypercube Sampling (LHS). LHS forces one sample per stratified bin along each axis, providing better coverage for the same number of samples.

| Random Monte Carlo | Latin Hypercube Sampling |
|:------------------:|:----------------------:|
| ![Random MC](https://raw.githubusercontent.com/BehroozBashokooh/HIIPAPP/main/docs/random_mc_example.png) | ![LHS](https://raw.githubusercontent.com/BehroozBashokooh/HIIPAPP/main/docs/lhs_example.png) |

*LHS guarantees one point per row and column strata (visualized by the uniform spread), reducing clustering and improving convergence for multidimensional problems.*

#### Gaussian Copula: Imposing Correlation
The Gaussian copula builds correlations in the latent normal space and maps the samples back to the requested marginals. Below we draw two marginals (Normal and Lognormal) with and without the copula correlation.

| Independent Marginals | Gaussian Copula (ρ = 0.7) |
|:---------------------:|:-------------------------:|
| ![Indep](https://raw.githubusercontent.com/BehroozBashokooh/HIIPAPP/main/docs/indep_example.png) | ![Copula](https://raw.githubusercontent.com/BehroozBashokooh/HIIPAPP/main/docs/copula_example.png) |

*Spearman correlation after the copula transform matches the intended dependency while preserving each marginal's shape.*

---
### Outputs & Interpretation
- **Histograms & CDFs** show the distribution of the active output with P10/P50/P90 markers (exceedance convention: P90 is low, P10 is high).
- **Spearman heat map** focuses on how each input ranks against the target output. When dependencies are disabled, it simplifies to an input-vs-output column; when enabled, the full matrix is displayed.
- **Tornado chart** reports the percentage change in P50 when each input is set to its P10 or P90 value (other variables remain sampled). Geometry inputs collapse into a single GRV bar when the scale factor is enabled to avoid double counting.
- **Results table** summarizes P10/P50/P90 and mean for every computed output (STOIIP/GIIP, reserves/recoverable, solution gas, condensate) using the units selected in the sidebar.

---
### Downloads & Reporting
- **CSV** contains every sampled input (post unit conversion) plus calculated outputs and unit-scaled columns that mirror the results table.
- **HTML report** embeds the charts you saw on screen along with the summary table so stakeholders can review without needing Streamlit.
- Both files are packaged into a single ZIP to avoid regenerating simulations when downloading.

---
### Tips & Good Practices
- Fractional inputs (NTG, phi, Sw, RF) are clipped to the physical [0, 1] interval after sampling.
- Use truncated normals or lognormals when bounds are known to prevent unrealistic extremes.
- When eliciting expert ranges, the PERT or Custom quantile options often provide more realistic tails than a triangular distribution.
- Re-run with different seeds to confirm stability of summary statistics if you work with smaller iteration counts.

---
### Version & Support
- **App version:** 1.2.0
- **Questions or feedback?** Use the sidebar email link to reach the maintainers.
        """