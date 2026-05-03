import matplotlib.pyplot as plt
from scipy.stats import linregress

from objects import *
from config import *


def plot_spectrum(spectrum, fit_angles, fit_intensities, lamp_name, save_dir="Plots/Spectra"):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(spectrum.angles, spectrum.intensities, color='red', marker='x', label='Data', zorder=3)
    ax.plot(fit_angles, fit_intensities, color='black', label='Malus fit')

    ax.set_xlabel('Analyser angle (θ°)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title(f'{lamp_name} — {spectrum.current}A spectrum')
    ax.legend()
    fig.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_dir}/{spectrum.current}A.png")
    plt.close(fig)


def plot_spectra_combined(spectra_data, lamp_name, save_dir="Plots/Spectra"):
    fig, ax = plt.subplots(figsize=(10, 6))

    currents = [s.current for s, _, _ in spectra_data]
    norm = plt.Normalize(vmin=min(currents), vmax=max(currents))
    cmap = plt.cm.coolwarm

    for spectrum, fit_angles, fit_intensities in spectra_data: 
        color = cmap(norm(spectrum.current))
        ax.scatter(spectrum.angles, spectrum.intensities, color=color, marker='x', zorder=3, s=20)
        ax.plot(fit_angles, fit_intensities, color=color, label=f'{spectrum.current}A')

    ax.set_xlabel('Analyser angle (θ°)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'{lamp_name} Spectra')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_dir}/{lamp_name} Combined.png", dpi=150)
    plt.close(fig)



def plot_faraday_rotation(lamp, B, thetas, sigmas, sigma_B, slope, intercept=0.0, save_dir="Plots/Faraday Rotations"):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(B, thetas, yerr=sigmas, xerr=sigma_B, fmt='none', capsize=5, color='red', zorder=3)
    ax.scatter(B, thetas, color='red', marker='x', label='Data', zorder=4)

    B_fit = np.linspace(B.min(), B.max(), 300)
    ax.plot(B_fit, np.degrees(slope * B_fit), color='steelblue', label='ODR linear fit')
    
    ax.set_xlabel('B (T)')
    ax.set_ylabel('θ (degrees)')
    ax.set_title(f'{lamp.name} — Faraday Rotation')
    ax.legend()
    fig.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_dir}/{lamp.name} Faraday Rotation.png")
    plt.close(fig)
