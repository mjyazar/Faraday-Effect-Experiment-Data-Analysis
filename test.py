import numpy as np
from pathlib import Path
from scipy.stats import linregress
from scipy.odr import ODR, Model, RealData

from config import *
from objects import Lamp
from plotting import plot_spectrum, plot_spectra_combined, plot_faraday_rotation

# ── helper functions ───────────────────────────────────────────────────────────

def compute_omega_0(V, omega):
    """Eigenfrequency from Verdet constant and light frequency."""
    A = (n**2 - 1) * q * omega**2 / (2 * n * m_e * c)
    return np.sqrt(omega**2 + A / V)

def sigma_omega_0(V, sigma_V, omega, omega_0):
    """Error on eigenfrequency."""
    A = (n**2 - 1) * q * omega**2 / (2 * n * m_e * c)
    return A / (2 * V**2 * omega_0) * sigma_V

def predict_verdet(omega, omega_0):
    """Predicted Verdet constant from eigenfrequency."""
    return (n**2 - 1) / (2 * n) * q * omega**2 / (m_e * c * (omega_0**2 - omega**2))

def compute_delta_n(theta_rad, wavelength):
    """Δn = nₗ - nᵣ from Faraday rotation angle in radians."""
    return abs(theta_rad) * wavelength / (np.pi * d)

def sigma_delta_n(sigma_theta_rad, wavelength):
    return wavelength / (np.pi * d) * sigma_theta_rad

def approximation_validity(omega, B_max, omega_0):
    """Check qωB / m(ω₀²-ω²) ≪ 1."""
    return q * omega * B_max / (m_e * (omega_0**2 - omega**2))

# ── data directories ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
NA_DIR   = BASE_DIR / "Data" / "Na"
HGCD_DIR = BASE_DIR / "Data" / "HgCd"

na   = Lamp("Na",   LAMBDA_NA,   NA_DIR)
hgcd = Lamp("HgCd", LAMBDA_HGCD, HGCD_DIR)

# ── store results for cross-lamp calculations ──────────────────────────────────

results = {}

for lamp in [na, hgcd]:
    print(f"\n{'═'*50}")
    print(f"  {lamp.name} lamp  (λ = {lamp.wavelength*1e9:.0f} nm)")
    print(f"{'═'*50}")

    currents, thetas, sigmas = lamp.faraday_rotation()

    # magnetic field and its uncertainty
    B       = ALPHA * currents
    sigma_B = np.sqrt((currents * sigma_ALPHA)**2 + (ALPHA * sigma_current)**2)

    # convert θ to radians for physics calculations
    thetas_rad = np.radians(thetas)
    sigmas_rad = np.radians(sigmas)

    # ── ODR fit: θ [rad] = V·d·B ──────────────────────────────────────────────
    def linear(params, x):
        return params[0] * x + params[1]

    p_init  = np.polyfit(B, thetas_rad, 1)
    odr_data  = RealData(B, thetas_rad, sx=sigma_B, sy=sigmas_rad)
    odr_model = Model(linear)
    odr_fit   = ODR(odr_data, odr_model, beta0=p_init)
    odr_result = odr_fit.run()

    slope       = odr_result.beta[0]       # V·d  [rad/T]
    sigma_slope = odr_result.sd_beta[0]

    # propagate d uncertainty into V
    V       = slope / d
    sigma_V = np.sqrt((sigma_slope / d)**2 + (slope * sigma_d / d**2)**2)

    print(f"Verdet constant:  V = {V:.4f} ± {sigma_V:.4f} rad/(T·m)")

    # ── eigenfrequency ─────────────────────────────────────────────────────────
    omega_0       = compute_omega_0(V, lamp.omega)
    d_omega_0     = sigma_omega_0(V, sigma_V, lamp.omega, omega_0)
    omega_0_THz   = omega_0 / 1e12
    d_omega_0_THz = d_omega_0 / 1e12
    print(f"Eigenfrequency:   ω₀ = ({omega_0_THz:.4f} ± {d_omega_0_THz:.4f}) × 10¹² rad/s")

    # ── Δn at maximum rotation ─────────────────────────────────────────────────
    idx_max       = np.argmax(np.abs(thetas_rad))
    theta_max_rad = thetas_rad[idx_max]
    sigma_max_rad = sigmas_rad[idx_max]
    I_max         = currents[idx_max]
    B_max         = B[idx_max]

    dn       = compute_delta_n(theta_max_rad, lamp.wavelength)
    sigma_dn = sigma_delta_n(sigma_max_rad,   lamp.wavelength)
    print(f"Δn at I={I_max}A:      Δn = ({dn:.6f} ± {sigma_dn:.6f})")

    # ── approximation validity ─────────────────────────────────────────────────
    epsilon = approximation_validity(lamp.omega, abs(B_max), omega_0)
    print(f"Approximation check: qωB/m(ω₀²-ω²) = {epsilon:.5f}  ({'✓ valid' if epsilon < 0.1 else '✗ questionable'})")

    results[lamp.name] = {
        'V': V, 'sigma_V': sigma_V,
        'omega_0': omega_0, 'sigma_omega_0': d_omega_0,
        'B': B, 'sigma_B': sigma_B,
        'thetas': thetas, 'sigmas': sigmas,
        'thetas_rad': thetas_rad, 'sigmas_rad': sigmas_rad,
        'currents': currents, 'fit': odr_result,
        'slope': slope,
    }

    # ── plots ──────────────────────────────────────────────────────────────────
    plot_faraday_rotation(lamp, B, thetas, sigmas, sigma_B, slope, d)

    spectra_data = []
    for current in SPECTRA_TO_PLOT[lamp.name]:
        spectrum = lamp.spectra[current]
        _, _, fit_angles, fit_intensities = spectrum.find_min()
        spectra_data.append((spectrum, fit_angles, fit_intensities))
        plot_spectrum(spectrum, fit_angles, fit_intensities, lamp.name)
    plot_spectra_combined(spectra_data, lamp.name)

# ── cross-lamp: predict V_HgCd from ω₀_Na ─────────────────────────────────────
print(f"\n{'═'*50}")
print("  Cross-lamp comparison")
print(f"{'═'*50}")

omega_0_Na     = results['Na']['omega_0']
sigma_omega_0_Na = results['Na']['sigma_omega_0']
omega_HgCd     = hgcd.omega

V_pred         = predict_verdet(omega_HgCd, omega_0_Na)

# error on V_pred from sigma_omega_0_Na
dV_pred_domega0 = -(n**2-1) * q * omega_HgCd**2 * 2 * omega_0_Na / \
                  (2 * n * m_e * c * (omega_0_Na**2 - omega_HgCd**2)**2)
sigma_V_pred   = abs(dV_pred_domega0) * sigma_omega_0_Na

V_meas     = results['HgCd']['V']
sigma_V_meas = results['HgCd']['sigma_V']

print(f"Measured  V_HgCd  = {V_meas:.4f} ± {sigma_V_meas:.4f} rad/(T·m)")
print(f"Predicted V_HgCd  = {V_pred:.4f} ± {sigma_V_pred:.4f} rad/(T·m)  (from ω₀_Na)")

# check consistency within combined uncertainty
discrepancy = abs(V_meas - V_pred)
combined_sigma = np.sqrt(sigma_V_meas**2 + sigma_V_pred**2)
n_sigma = discrepancy / combined_sigma
print(f"Discrepancy: {discrepancy:.4f} rad/(T·m)  =  {n_sigma:.1f}σ")
