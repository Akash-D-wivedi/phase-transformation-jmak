## 1. Phase Transformations in Steel

Steel is an iron–carbon alloy whose microstructure evolves dramatically upon cooling from the austenitizing temperature (typically 900 °C). The face-centered cubic (FCC) austenite phase transforms into several lower-temperature phases depending on the thermal path:

* **Ferrite (α‑Fe):** Body‑centered cubic (BCC) iron with low carbon solubility. Forms by diffusion of carbon away from growing ferrite nuclei. Typical isothermal holds: 700–800 °C.
* **Pearlite:** A lamellar mixture of ferrite and cementite (Fe₃C) that forms by cooperative diffusion of carbon. Observed at intermediate temperatures (550–700 °C).
* **Bainite:** A fine microconstituent of ferrite and cementite that forms at lower temperatures (300–550 °C). Consists of upper bainite (higher temperature, coarser) and lower bainite (lower temperature, finer).

The **Time–Temperature–Transformation (TTT) diagram** maps the start and finish times of these transformations at different isothermal temperatures and guides heat‑treatment design in practice.

---

## 2. The Johnson–Mehl–Avrami–Kolmogorov (JMAK) Equation

The JMAK model describes kinetics of phase transformations controlled by nucleation and growth:

$$
f(t) = y_{\max}\bigl[1 - \exp\bigl(- (k\,t)^{n}\bigr)\bigr]
$$

* $f(t)$: Transformed fraction at time $t$.
* $y_{\max}$: Asymptotic maximum fraction (≤ 1) if transformation saturates before completion.
* $k$: Rate constant (1/s), often following Arrhenius behavior.
* $n$: Avrami exponent, dependent on nucleation rate and growth dimensionality.

**Interpretation of the Avrami exponent $n$:**

* $n \approx 1$: Site‑saturated nucleation with one‑dimensional growth.
* $n \approx 2$: Continuous nucleation with two‑dimensional growth.
* $n \approx 3$: Continuous nucleation with three‑dimensional growth.

---

## 3. Isothermal Kinetics Fitting Procedure

1. **Experimental anchor points:** Extract $t_{start}$ (0 %), $t_{10}$ (10 %), $t_{90}$ (90 %) from TTT or dilatometry data at each isothermal temperature.
2. **Global or per‑isotherm $n$:** With only three points, it is common to fix $n$ (literature value) and fit $k$ by non-linear least squares. With dense data (> 6 points), fit both $n$ and $k$.
3. **Arrhenius analysis:** The temperature dependence of the rate constant follows:

   $$
   \k(T) = \k_0 \, \exp\bigl[(-\frac{Q}{R\,T}\Bigr]
   $$

   * $k_0$: Pre‑exponential factor.
   * $Q$: Activation energy (J/mol).
   * $R$: Universal gas constant (8.314 J/mol·K).

   Plot $\ln k$ vs. $1/T$ or fit directly to extract $k_0$ and $Q$.

---

## 4. Uncertainty Quantification Methods

Accurate modeling requires estimates of confidence:

* **Bootstrapping:** Resample the 0/10/90 % points with added timing noise to generate distributions of fitted $k$ and $n$, then propagate to $f(t)$.
* **Gaussian Process Regression:** Fit a GP to $\ln k(T)$ data, yielding a predictive mean and standard deviation. Transform back to parameter space and compute 95 % confidence bands for $f(t)$.

These uncertainty bands illustrate the precision of kinetic predictions and guide process‑control decisions.
