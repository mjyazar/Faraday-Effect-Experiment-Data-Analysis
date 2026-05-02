import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
from config import *


class Spectrum:
    def __init__(self, current: int, filepath):
        data = np.loadtxt(filepath, delimiter=',')
        self.angles = data[:, 0]
        self.intensities = data[:, 1]
        self.current = current
        self.min_angle = self.angles[np.argmin(self.intensities)]

    def find_min(self):
        def malus_law(theta, I0, phi, offset):

            return I0 * np.cos(np.radians(theta - phi))**2 + offset

        # guess initial parameters for optimising curve fitting
        p0 = [
            self.intensities.max(),
            self.angles[self.intensities.argmin()] - 90,
            self.intensities.min()
        ]
        param, param_cov = curve_fit(malus_law, self.angles, self.intensities, p0 = p0)

        fit_angles      = np.linspace(self.angles.min(), self.angles.max(), 300)
        fit_intensities = malus_law(fit_angles, *param)
        
        min_angle = param[1] + 90
        dmin_angle = np.sqrt(param_cov[1, 1])
        #print(f"Current: {self.current}A")
        #print(f"Minimum angle: {min_angle:.2f} ± {dmin_angle:.2f}, {self.angles[np.argmin(self.intensities)]}\n")

        return min_angle, dmin_angle, fit_angles, fit_intensities


class Lamp:
    def __init__(self, name, wavelength, data_directory):
        self.name = name
        self.wavelength = wavelength
        self.data_directory = data_directory
        self.zero_field = None
        self.spectra = {}
        self._load_spectra()

    
    def _load_spectra(self):

        for file_path in sorted(Path(self.data_directory).glob("*.csv")):
            print(f"Loading file: {file_path}")

            if file_path.stem == "No Field":
                self.zero_field = Spectrum(0, file_path)

            else:
                current = int(file_path.stem.split('A')[0])
                print(f"Extracted current: {current}A\n")

                s = Spectrum(current, file_path)
                self.spectra[current] = s
    
    def faraday_rotation(self):
        phi0, dphi0, _, _ = self.zero_field.find_min()

        currents, thetas, sigmas = [], [], []
        for current, spectrum in self.spectra.items():
            phi, dphi, _, _ = spectrum.find_min()
            currents.append(current)
            thetas.append(phi - phi0)
            sigmas.append(np.sqrt(dphi**2 + dphi0**2))

        return np.array(currents), np.array(thetas), np.array(sigmas)

