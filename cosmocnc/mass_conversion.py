"""NumPy mirror of cosmocnc_jax.mass_conversion (Castro23 HMF support subset).

Function-for-function port of the pieces needed for the M_200c -> M_vir
conversion (sigma-based B13 c_vir + BN98 virial overdensity + NFW Newton).
Kept numerically identical to the JAX implementation: same constants, same
damped-Newton scheme (n_iter=20, central-diff eps=1e-3, step clip +-1), same
interpolation semantics (np.interp == jnp.interp).

Reference: cosmocnc_jax/mass_conversion.py (which itself mirrors classy_sz's
mDEL_to_mDELprime path); Castro et al. 2023 (arXiv:2208.02174) uses virial
masses with the BN98 spherical-collapse overdensity.
"""

import numpy as np


def _nfw_m(y):
    """NFW dimensionless enclosed mass: m(y) = ln(1+y) - y/(1+y)."""
    return np.log1p(y) - y / (1.0 + y)


def growth_factor_carroll_press_turner(z, Om0, OL0):
    """Carroll-Press-Turner 1992 fitting formula for D(z), normalised so D(0)=1.

    Identical to the cosmocnc_jax version (radiation ignored, subpercent z<5).
    """
    g0 = 2.5 * Om0 / (Om0**(4./7.) - OL0 + (1. + Om0 / 2.) * (1. + OL0 / 70.))
    a = 1.0 / (1.0 + z)
    Ez2 = Om0 / a**3 + OL0
    Om_z = (Om0 / a**3) / Ez2
    OL_z = OL0 / Ez2
    g_z = 2.5 * Om_z / (Om_z**(4./7.) - OL_z + (1. + Om_z / 2.) * (1. + OL_z / 70.))
    return g_z * a / g0


# classy_sz's spherical-collapse delta_c (input.c:7374)
DELTA_C_SZ = (3.0 / 20.0) * (12.0 * np.pi)**(2.0 / 3.0)


def b13_cvir_sigma_based(sigma, D_z, delta_c=DELTA_C_SZ):
    """Bhattacharya et al. 2013 c_vir with sigma-based peak height
    (classy_sz evaluate_cvir_of_mvir, concentration_parameter==6)."""
    nu = delta_c / sigma
    return D_z**0.9 * 7.7 * nu**(-0.29)


def delta_c_virial(Om_z):
    """Bryan & Norman 1998 virial overdensity wrt critical:
    18 pi^2 + 82 x - 39 x^2, x = Om(z) - 1."""
    x = Om_z - 1.0
    return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2


def solve_M_vir_from_M_200c(M_200c, rho_c_z, Om_z, D_z,
                            logM_grid_for_sigma, sigma_grid_at_z,
                            delta_c_sc=DELTA_C_SZ, n_iter=20):
    """Vectorised damped-Newton solve for M_vir given M_200c (BN98 virial,
    sigma-based B13 c_vir). Mirror of cosmocnc_jax's _m200c_to_mvir_one /
    _solve_M_vir_from_M_DEL (same scheme: central-diff derivative in
    log M_vir with eps=1e-3, step clipped to +-1, init at log M_200c).

    Args:
        M_200c: array of M_200c in physical Msun.
        rho_c_z, Om_z, D_z: scalars at this z.
        logM_grid_for_sigma: (n_logM,) ln(M_phys) grid where sigma is tabulated.
        sigma_grid_at_z: (n_logM,) sigma values at this z.

    Returns:
        M_vir array, same shape as M_200c.
    """
    M_200c = np.asarray(M_200c, dtype=np.float64)
    d_c_vir = delta_c_virial(Om_z)
    R_DEL = (3.0 * M_200c / (4.0 * np.pi * 200.0 * rho_c_z))**(1.0/3.0)

    def f_of_logM(lm):
        Mv = np.exp(lm)
        sv = np.interp(lm, logM_grid_for_sigma, sigma_grid_at_z)
        cv = b13_cvir_sigma_based(sv, D_z, delta_c_sc)
        Rv = (3.0 * Mv / (4.0 * np.pi * d_c_vir * rho_c_z))**(1.0/3.0)
        rs = Rv / cv
        C = R_DEL / rs
        return M_200c / Mv - _nfw_m(C) / _nfw_m(cv)

    eps = 1e-3
    log_M_vir = np.log(M_200c)
    for _ in range(n_iter):
        f_now = f_of_logM(log_M_vir)
        f_plus = f_of_logM(log_M_vir + eps)
        f_minus = f_of_logM(log_M_vir - eps)
        f_prime = (f_plus - f_minus) / (2.0 * eps)
        delta = np.clip(f_now / f_prime, -1.0, 1.0)
        log_M_vir = log_M_vir - delta

    return np.exp(log_M_vir)
