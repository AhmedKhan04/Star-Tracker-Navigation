# Star Tracker Navigation
**Photometry & Star Navigation Toolkit**

This repository implements a **hybrid stellar navigation system**
capable of estimating spacecraft **position and clock offset** using
pulsating stars and stellar light‑curve timing.

Unlike traditional star trackers that rely purely on star geometry, this
system uses **time delays in stellar brightness variations** to infer
spacecraft state. The concept is analogous to **pulsar navigation**, but
implemented using **optical variable stars** such as Delta Scuti stars.

The system supports both:

-   **Synthetic stars** for simulation
-   **Real observational light curves** from telescope data

The navigation solution is obtained by measuring **time offsets between
observed and reference stellar signals** and solving a geometric system
relating those delays to spacecraft position.

This work was done in collaboration with the Laboratory of Advanced Space System at Illinois, Astrodynamics and Planetary Exploration research group at **UIUC** as well as the **Champaign-Urbana Astronomical Society**.

------------------------------------------------------------------------

# 🌌 Overview:  Navigation Principle

Pulsating stars produce periodic brightness signals:

F(t)

Because the spacecraft is not located at the reference observation
location, the signal arrives with a **geometric light‑travel delay**.

Δt = (û · r) / c

Where:

-   û : unit vector toward the star\
-   r : spacecraft position vector\
-   c : speed of light

The spacecraft clock may also contain an unknown offset:

t_clock

Therefore the measured delay becomes:

Δt_obs = t_clock + (û · r) / c

By observing multiple stars and comparing to onboard models, the spacecraft can estimate both:

-   its **position**
-   its **clock offset**

If you’re working on **satellite navigation, star trackers, or light curve analysis, Star Tracker Navigation offers a modular foundation.**


> **Note:**  
> Some data files, outputs, and logs are **locally hosted** and intentionally excluded from this repository as indicated in the `.gitignore`.  
> These include large datasets, FITS images, and test results not suitable for public hosting.


# 🎲 Monte Carlo Simulation

The function:

monte_carlo_simulation()

runs repeated navigation trials with random noise to evaluate
performance.

Statistics reported include:

-   position error
-   clock error
-   residual fitting error
-   success rate

This allows comparison with published stellar navigation performance
metrics.

------------------------------------------------------------------------

# ⚙️ Example Workflow

Typical execution sequence:

1.  Load calibration frames
2.  Compile stellar light curves
3.  Build star models
4.  Simulate spacecraft observations
5.  Estimate stellar delays
6.  Solve spacecraft position and clock offset
7.  Run Monte‑Carlo analysis

The main entry point of the program is:

main()

in `Navigation_algo_hybrid.py`

------------------------------------------------------------------------

## 🧠 Getting Started

### Prerequisites
You’ll need:
```bash
Python >= 3.8
numpy
scipy
astropy
pandas
matplotlib
lightkurve   # optional, for testing with the lightkurve library

This navigation method is relevant for:

-   deep‑space spacecraft
-   autonomous probes
-   GPS‑denied environments
-   optical pulsar‑style navigation

The approach demonstrates how **stellar variability can act as natural
navigation beacons**.

------------------------------------------------------------------------

## Author

Ahmed Khan

Undergraduate research in **Spacecraft GNC**.
