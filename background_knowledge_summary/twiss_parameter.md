# Twiss Parameters

A complete description of a beam's transverse phase-space distribution at any
location $s$ along the beamline. Together $(\beta, \alpha, \gamma, \epsilon)$
fully characterize the beam ellipse in $(x, x')$ space; one set per transverse
plane (x, y), and a separate set for the longitudinal plane (z, $\delta$).

## Phase-space ellipse (the unifying picture)

At any $s$, the beam particles occupy an ellipse in $(x, x')$ space described by
the **Courant-Snyder invariant**:

$$
\gamma(s) x^2 + 2 \alpha(s) x x' + \beta(s) {x'}^2 = \epsilon
$$

- The **ellipse area** is $\pi \epsilon$ — emittance, conserved (adiabatically) along the beamline
- The **ellipse shape** at each $s$ is determined by $(\beta, \alpha, \gamma)$
- These three are not independent — they satisfy the constraint:
$$
\beta \gamma - \alpha^2 = 1 \quad \Longrightarrow \quad \gamma = \frac{1 + \alpha^2}{\beta}
$$

So Twiss has **two free degrees of freedom** at each $s$ ($\beta, \alpha$), plus
the conserved emittance $\epsilon$. As particles propagate through magnets, the
ellipse rotates and stretches — but its area $\pi \epsilon$ stays the same
(Liouville's theorem).

| Twiss | Geometric meaning on the ellipse | Units |
|:-----:|:---------------------------------|:------|
| $\beta$ | Half-width along $x$-axis is $\sqrt{\beta \epsilon}$ | m |
| $\alpha$ | Tilt of the ellipse (slope of waist crossing) | dimensionless |
| $\gamma$ | Half-height along $x'$-axis is $\sqrt{\gamma \epsilon}$ | rad/m |
| $\epsilon$ | Area of ellipse / π | m·rad |

## beta $\beta$ — beam size function

$\beta_x(s)$ is the **squared beam size**, normalized by emittance. It tells you
how wide the beam is at position $s$ in the $x$ direction:

$$
\sigma_x(s) = \sqrt{\beta_x(s) \cdot \epsilon_x}
$$

where $\sigma_x$ is the RMS beam size (transverse half-width containing ~68%
of particles). Same in $y$.

**Physical interpretation**:
- $\beta$ **small** → beam is **narrow** at that point (focused down)
- $\beta$ **large** → beam is **wide** (defocused, or far from a waist)
- $\beta$ has units of meters and is always positive in any physically meaningful lattice

**Why "beta function" and not just "beam size"?**
Because $\beta(s)$ is an intrinsic lattice property — it depends only on the
quadrupoles and drifts, not on the beam injected. The actual beam size
$\sigma_x$ varies with both the lattice ($\beta$) and the beam quality
($\epsilon$). A clean lattice has a well-defined $\beta(s)$ regardless of
which beam you push through it.

---

## alpha $\alpha$ — convergence/divergence indicator

$\alpha_x(s)$ measures how $\beta_x$ is changing along the beamline:

$$
\alpha_x(s) = -\frac{1}{2} \frac{d\beta_x}{ds}
$$

Three regimes:

| Value | Meaning | What's happening |
|:------|:--------|:-----------------|
| $\alpha > 0$ | $\beta$ decreasing | Beam is **converging** toward a waist |
| $\alpha = 0$ | $\beta$ at local extremum | At a **waist** (or anti-waist) |
| $\alpha < 0$ | $\beta$ increasing | Beam is **diverging** from a waist |

Geometrically on the phase-space ellipse:
- $\alpha = 0$ → ellipse axes aligned with $(x, x')$ axes
- $\alpha > 0$ → ellipse tilts forward (positive correlation between $x$ and $x'$ for converging beam... actually negative, see below)
- $\alpha < 0$ → ellipse tilts backward

A simple way to remember: **at a waist ($\alpha = 0$), the beam reaches its
minimum size** and starts diverging again. This is why FEL operation prefers
$\alpha = 0$ at the undulator entrance — the beam should be at (or near) its
smallest, most stable size as it enters the radiation section.

---

## gamma $\gamma$ — angular spread function

$\gamma_x(s)$ is the **squared angular divergence**, normalized by emittance:

$$
\sigma_{x'}(s) = \sqrt{\gamma_x(s) \cdot \epsilon_x}
$$

where $\sigma_{x'}$ is the RMS angle spread (in radians) — how much the
particles' trajectories diverge from the centerline at position $s$.

Strictly determined by $\beta$ and $\alpha$ via the constraint $\beta\gamma - \alpha^2 = 1$:

$$
\gamma_x = \frac{1 + \alpha_x^2}{\beta_x}
$$

So $\gamma$ doesn't carry independent information — but it's useful as a
diagnostic, especially at waists where $\beta$ is small and $\sigma_x$ shrinks
but $\sigma_{x'}$ grows. This trade-off is the **uncertainty-principle analog**
in beam optics: focused beams diverge fast; collimated beams are wide.

---

## emittance $\epsilon$ — beam quality

$\epsilon$ is the **phase-space area** occupied by the beam (divided by $\pi$):

$$
\epsilon_x^2 = \langle x^2 \rangle \langle {x'}^2 \rangle - \langle x x' \rangle^2
$$

(All averages over the particle distribution.) Units: $\pi \cdot$mm$\cdot$mrad
(or m$\cdot$rad in SI).

Physical interpretation:
- **Small ε** → beam is "tight" in both position and angle → high brightness
- **Large ε** → particles are spread out in $(x, x')$ → low brightness
- $\epsilon$ is **conserved** as the beam moves through linear optics (drifts, quads, dipoles)
  — this is Liouville's theorem applied to charged-particle beams

**Geometric emittance vs normalized emittance**:

$$
\epsilon_n = \beta_\text{rel} \gamma_\text{rel} \cdot \epsilon_\text{geom}
$$

(Careful: these $\beta_\text{rel}, \gamma_\text{rel}$ are the relativistic factors,
**not** the Twiss parameters above — unfortunate naming collision.)

- $\epsilon_\text{geom}$ shrinks as beam is accelerated (high energy → tighter cone)
- $\epsilon_n$ is invariant under acceleration — the true "intrinsic brightness"

In your FELsim code, `cal_twiss` returns geometric emittance in π·mm·mrad.

---

## Target Twiss at Undulator Entrance

What do you actually want $\beta, \alpha, \epsilon$ to be at the undulator
entrance? This is the FEL matching problem.

### Vertical: $\beta_y$ is **forced** by undulator physics

Most planar undulators have a vertical magnetic gradient that produces
**natural vertical focusing** — the undulator acts like a continuous weak
focusing lens in $y$. To avoid betatron oscillations that beat against the
undulator's own focusing period, the injected beam must match this natural
focusing exactly:

$$
\beta_{y,\text{match}} = \frac{\gamma_\text{rel}}{K \cdot k_u} = \frac{\gamma_\text{rel} \lambda_u}{2\pi K}
$$

For Mark V FEL: $\gamma_\text{rel} = 79.3$ (40 MeV), $K = 1.2$, $\lambda_u = 2.3$ cm:

$$
\beta_{y,\text{match}} = \frac{79.3 \times 0.023}{2\pi \times 1.2} \approx 0.242 \text{ m}
$$

**This is a hard requirement** — mismatching it causes beam size oscillations
along the undulator, which kills FEL gain because the electron beam–photon
beam overlap fluctuates.

And **$\alpha_y = 0$** at the entrance, so the beam is at its waist exactly when
it enters the natural-focusing regime. (Otherwise the waist forms inside the
undulator, breaking matching.)

### Horizontal: $\beta_x$ is a **design choice**

Planar undulators do **not** provide natural horizontal focusing — the field is
uniform in $x$. So $\beta_x$ along the undulator just behaves like in a long
drift: it grows quadratically. To minimize the beam size averaged over the
undulator length, designers typically choose a waist in the middle:

$$
\beta_x(s) = \beta_{x,\text{waist}} + \frac{(s - s_\text{waist})^2}{\beta_{x,\text{waist}}}
$$

Optimal $\beta_{x,\text{waist}}$ is on the order of $L_u / 2$, where $L_u$ is
the undulator length. For Mark V (a few meters):

$$
\beta_{x,\text{match}} \approx 1\text{–}3 \text{ m}, \quad \alpha_{x,\text{match}} \approx 0.5
$$

In your notebook you have:
- $\beta_{x,m} = 1.4$ m (waist placed slightly upstream so beam shrinks into undulator)
- $\alpha_{x,m} = 0.47$ (mild convergence at entrance — waist will form ~0.66 m downstream)

### Longitudinal (z): is there a Twiss requirement?

**Generally no — in the rigorous "Twiss" sense.** Longitudinal phase space
$(z, \delta)$ where $\delta = \Delta p/p$ is the relative momentum deviation,
has its own Twiss-like parameters $(\beta_z, \alpha_z, \epsilon_z)$, and FELsim's
PyTorch-version `cal_twiss` now reports them. But the FEL gain process cares
about different longitudinal quantities:

1. **Bunch length** $\sigma_z$ — should be short enough that the bunch fits
   within a few "slippage lengths" (the distance the photon overruns the electron
   per undulator period). Typical: $\sigma_z \sim 100$ μm – few mm depending on FEL design.

2. **Energy spread** $\sigma_\delta = \sigma_E / E$ — must be **smaller than the
   FEL Pierce parameter** $\rho_\text{FEL}$ (a dimensionless gain parameter,
   typically $10^{-3}$ to $10^{-4}$ for IR/UV FELs). If energy spread exceeds
   $\rho_\text{FEL}$, electrons at different energies oscillate at different
   undulator frequencies and lose coherence — FEL gain drops to zero.

3. **Energy chirp** $d\delta/dz$ — correlation between longitudinal position and
   energy. Useful in bunch compression upstream, but ideally zero at the
   undulator entrance.

So instead of $(\beta_z, \alpha_z)$ matching to a target, the longitudinal
requirements are:

| Quantity | Target | Hard limit |
|:---------|:-------|:-----------|
| $\sigma_z$ | "Short enough" | Set by slippage; not a sharp threshold |
| $\sigma_\delta$ | As small as possible | $\sigma_\delta < \rho_\text{FEL} \sim 10^{-3}$ |
| Energy chirp | Zero | $|d\delta/dz| \cdot \sigma_z < \rho_\text{FEL}$ |

Your Scenario A/B/C objectives only constrain transverse Twiss because FELsim
currently assumes the upstream linac+chicane already delivers acceptable
longitudinal parameters. The longitudinal subsystem (RF phase, chicane R56)
would be a separate optimization scenario — not yet modeled in FELsim.

### Summary table for Mark V FEL @ 40 MeV

| Quantity | Target | Source |
|:---------|:-------|:-------|
| $\beta_y$ (entrance) | $\gamma_\text{rel} \lambda_u / (2\pi K) \approx 0.242$ m | Natural focusing physics |
| $\alpha_y$ (entrance) | $0$ | Waist at undulator start |
| $\beta_x$ (entrance) | $\sim 1.4$ m | Design choice (waist mid-undulator) |
| $\alpha_x$ (entrance) | $\sim 0.47$ | Beam slightly converging into undulator |
| $\epsilon_x = \epsilon_y$ (normalized) | $\sim 1$–$10$ π·mm·mrad | Set by gun emittance, conserved |
| $\sigma_z$ | $< $ few mm | Set by chicane / RF compression |
| $\sigma_\delta$ | $< 10^{-3}$ | FEL gain bandwidth |
| Energy chirp | $\approx 0$ | Compensated upstream |

Notes:
- Transverse: hard physics requirement for $\beta_y$, designer's choice for $\beta_x$
- Longitudinal: no Twiss-style matching; instead bunch length and energy spread thresholds
- Your FELsim scenarios optimize only the transverse part (5 of the 8 quantities above)


## phi $\phi$ — phase advance / ellipse orientation

`cal_twiss` computes:

$$
\phi = \frac{1}{2}\arctan\!\left(\frac{2\alpha}{\gamma - \beta}\right)
$$

returned in degrees.

**What it measures**: the **orientation of the phase-space ellipse** — i.e. how
much the $(x, x')$ ellipse is rotated relative to the coordinate axes. It comes
from diagonalizing the ellipse's quadratic form (the rotation angle that aligns
the ellipse's principal axes with the $(x, x')$ axes).

**Relationship to $\alpha$**:
- $\alpha = 0$ → ellipse axes aligned with $(x, x')$ → $\phi = 0$ (beam at a waist)
- $\alpha \neq 0$ → ellipse tilted → $\phi \neq 0$

So $\phi$ and $\alpha$ both encode the ellipse tilt, but differently: $\alpha$ is
the Twiss tilt parameter, $\phi$ is the literal **geometric rotation angle** of
the ellipse in degrees. $\phi$ is a derived diagnostic — it carries no
information beyond $(\alpha, \beta, \gamma)$.

**Two distinct meanings of "phase" — don't confuse them**:
- The $\phi$ here is the **instantaneous ellipse orientation** at one location $s$
  (a static geometric angle).
- The **betatron phase advance** $\mu(s) = \int_0^s ds'/\beta(s')$ is a different
  quantity — the accumulated oscillation phase as the beam travels, which governs
  how many betatron oscillations fit in the lattice. `cal_twiss` returns the
  former (ellipse orientation), **not** the integrated betatron phase. Despite the
  variable name `phi`, this is the per-location ellipse tilt, computed purely from
  the local second moments.

**Practical use**: mostly a diagnostic. When you optimize $\alpha = 0$, you are
implicitly driving $\phi \to 0$ as well (upright ellipse = waist). You won't
usually target $\phi$ directly.

---

## envelope — RMS beam size in physical units

`cal_twiss` doesn't return this directly; `ebeam.envelope` computes it from the
Twiss output:

$$
\text{envelope} = 10^{3} \cdot \sqrt{\epsilon \cdot \beta}
$$

(with $\epsilon$ converted to SI via the $10^{-6}$ factor in the code, and the
$10^{3}$ putting the result in mm).

**What it measures**: the **physical RMS beam size** $\sigma$ at that location —
the actual half-width of the beam in millimeters, i.e. how big the beam *really
is* on a screen.

This is exactly the $\sigma_x = \sqrt{\beta_x \epsilon_x}$ relation from the
$\beta$ section, packaged as a directly-usable number:

$$
\sigma_u = \sqrt{\beta_u \, \epsilon_u}
$$

**Why it's useful as a separate quantity**:
- $\beta$ alone is a *lattice* property (doesn't know the beam's emittance)
- $\epsilon$ alone is a *beam* property (doesn't know the focusing)
- **envelope combines both** into the thing you physically observe — the beam
  spot size. If you want "how many mm wide is the beam here," this is the number.

**Practical use**: aperture / clearance checks (does the beam fit through the
vacuum chamber, the undulator gap?), and beam-size matching targets stated in
physical units rather than abstract Twiss. It's $\beta$ and $\epsilon$ expressed
as something you can measure on a profile monitor.

---

## dispersion $D$ — position shift per unit energy deviation

`cal_twiss` computes (for the transverse planes only):

$$
D = \frac{\langle u\,\delta \rangle}{\langle \delta^2 \rangle}
= \frac{\mathrm{Cov}(u, \delta)}{\mathrm{Var}(\delta)}
$$

returned in mm. (`ebeam.disper` reads this $D$ column out of the Twiss table.)

**What it measures**: how much the **closed orbit shifts per unit relative energy
deviation** $\delta = \Delta p / p$. Physically — particles with different
energies are bent by different amounts in dipoles (and kicked differently by
off-center trajectories in quads), so a particle with energy offset $\delta$ sits
at transverse position $D \cdot \delta$ away from the reference orbit:

$$
x(\delta) = x_\beta + D\,\delta
$$

where $x_\beta$ is the pure betatron part and $D\delta$ is the energy-dependent
offset.

**Why it matters / why `cal_twiss` corrects for it**:
- Dispersion **inflates the apparent beam size**: a spread of energies $\sigma_\delta$
  smears the beam transversely by $D \cdot \sigma_\delta$, even if the true
  betatron emittance is small. The measured $\langle x^2 \rangle$ contains both
  the real beam and this dispersive smearing.
- This is precisely why `cal_twiss` does the **dispersion correction** (step 3):
  it subtracts $D^2 \sigma_\delta$ from the variance before computing emittance,
  so the reported $\epsilon$ is the *true* betatron emittance, not contaminated by
  energy spread. Without this, a dispersive region would report a falsely large
  emittance.

**$D$ is computed only for x and y** (transverse). The longitudinal plane (z)
keeps $D = 0$ by construction — dispersion is defined as transverse-position-vs-
energy, which doesn't apply to the energy axis itself.

**Practical use**:
- At the undulator you usually want $D \approx 0$ (dispersion-free) — energy
  spread shouldn't blow up the transverse beam size in the radiation section.
- In a chicane/bunch-compressor, large $D$ (specifically $R_{56}$, its
  longitudinal cousin) is *desired* — that's how energy chirp gets converted to
  longitudinal compression. But at the FEL undulator, $D$ should close to zero.

---

## Where each lives in `cal_twiss`

Quick map of which line in `cal_twiss` produces what:

| Quantity | Source in `cal_twiss` | Returned via |
|:---------|:----------------------|:-------------|
| $\epsilon$ | `epsilon = sqrt(var_corr*var_prime_corr - covar_corr²)` | `ebeam.epsilon` |
| $\alpha$ | `alpha = -covar_corr / epsilon` | `ebeam.alpha` |
| $\beta$ | `beta = var_corr / epsilon` | `ebeam.beta` |
| $\gamma$ | `gamma = var_prime_corr / epsilon` | `ebeam.gamma` |
| $D, D'$ | `D[:2] = cov[idx[:2],5] / sigma_delta` | `ebeam.disper` |
| $\phi$ | `phi = 0.5*atan2(2α, γ-β)` | `ebeam.phi` |
| envelope | $10^3\sqrt{\epsilon\beta}$ (in `ebeam.envelope`, not `cal_twiss`) | `ebeam.envelope` |