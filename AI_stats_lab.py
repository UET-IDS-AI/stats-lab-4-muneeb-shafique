"""
AI Stats Lab
Random Variables and Distributions
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():
    """
    X ~ Exp(1)
    CDF: F(x) = 1 - e^{-x}

    P(X > 5)        = e^{-5}
    P(X < 5)        = 1 - e^{-5}
    P(3 < X < 7)    = e^{-3} - e^{-7}
    """

    # STEP 1 — Analytic
    analytic_gt5     = math.exp(-5)
    analytic_lt5     = 1 - math.exp(-5)
    analytic_interval = math.exp(-3) - math.exp(-7)

    # STEP 2 — Simulate
    np.random.seed(42)
    samples = np.random.exponential(scale=1.0, size=100_000)

    # STEP 3 — Estimate P(X > 5)
    simulated_gt5 = np.mean(samples > 5)

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5


# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():
    """
    f(x) = 2x * e^{-x^2}  for x >= 0

    Non-negativity: x >= 0  =>  f(x) >= 0  ✓
    Integral: substitution u = x^2, du = 2x dx
              => integral_0^inf e^{-u} du = 1  ✓
    """

    # STEP 1 — Non-negativity holds for x >= 0

    # STEP 2 — Compute integral numerically
    f = lambda x: 2 * x * math.exp(-x**2)
    integral_value, _ = quad(f, 0, np.inf)

    # STEP 3 — Check validity
    is_valid_pdf = abs(integral_value - 1.0) < 1e-3

    # STEP 4 — Plot on [0, 3]
    xs = np.linspace(0, 3, 300)
    ys = 2 * xs * np.exp(-xs**2)

    plt.figure()
    plt.plot(xs, ys, label=r'$f(x)=2xe^{-x^2}$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('PDF Validation Plot')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pdf_plot.png')
    plt.close()

    return integral_value, is_valid_pdf


# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():
    """
    X ~ Exp(1)

    P(X > 5)      = e^{-5}
    P(1 < X < 3)  = e^{-1} - e^{-3}
    """

    # STEP 1 — Analytic
    analytic_gt5     = math.exp(-5)
    analytic_interval = math.exp(-1) - math.exp(-3)

    # STEP 2 — Simulate
    np.random.seed(0)
    samples = np.random.exponential(scale=1.0, size=100_000)

    # STEP 3 — Estimate
    simulated_gt5     = np.mean(samples > 5)
    simulated_interval = np.mean((samples > 1) & (samples < 3))

    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
    """
    X ~ N(10, 2^2)

    Standardize: Z = (X - 10) / 2

    P(X <= 12) = P(Z <= 1)  = Phi(1)
    P(8 < X < 12) = P(-1 < Z < 1) = Phi(1) - Phi(-1)
    """

    # STEP 2 — Analytic via standard normal CDF
    analytic_le12    = norm.cdf(1)               # P(Z <= 1)
    analytic_interval = norm.cdf(1) - norm.cdf(-1)  # P(-1 < Z < 1)

    # STEP 3 — Simulate
    np.random.seed(7)
    samples = np.random.normal(loc=10, scale=2, size=100_000)

    # STEP 4 — Estimate
    simulated_le12    = np.mean(samples <= 12)
    simulated_interval = np.mean((samples > 8) & (samples < 12))

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval