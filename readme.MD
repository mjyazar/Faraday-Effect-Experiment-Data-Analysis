# Faraday Effect — Data Analysis

This repository contains the data analysis pipeline for the lab experiment measuring the **Faraday effect** in glass: the rotation of linearly polarised light when it passes through a medium in the presence of a longitudinal magnetic field.

The experiment was done  at several magnetic field strengths with two spectral lamps: sodium (Na, 589 nm) and mercury-cadmium (HgCd, 546 nm). From the rotation angles we extract the **Verdet constant** of the glass, and from that the **eigenfrequency** of the bound electrons.

---

## What the code does

1. Loads polarimeter spectra (intensity vs. analyser angle) for each lamp and current setting.
2. Fits each spectrum to Malus's law to find the extinction angle, i.e. where the transmitted intensity is minimum.
3. Computes the Faraday rotation angle for each field value.
4. Fits θ vs. B with ODR (orthogonal distance regression, accounting for errors on both axes) to get the slope V·d, from which the Verdet constant V is extracted.
5. Derives the electron eigenfrequency ω₀ from V and checks consistency between the two lamps.

---

## Structure

```
.
├── main.py        # top-level script
├── objects.py     # Spectrum and Lamp classes (data loading + angle extraction)
├── plotting.py    # all plot functions
├── config.py      # physical constants and experimental parameters
├── Data/
│   ├── Na/        # Na lamp spectra (.csv, one per current)
│   └── HgCd/      # HgCd lamp spectra (.csv, one per current)
└── Plots/
    ├── Faraday Rotations/   # θ vs B plots with ODR fit
    └── Spectra/             # individual and combined spectra plots
```

---

## Plots

### Faraday Rotation — Na (589 nm)
![Na Faraday Rotation](<Plots/Faraday Rotations/Na Faraday Rotation.png>)

### Faraday Rotation — HgCd (546 nm)
![HgCd Faraday Rotation](<Plots/Faraday Rotations/HgCd Faraday Rotation.png>)

### Spectra — Na
![Na Combined](<Plots/Spectra/Na Combined.png>)

### Spectra — HgCd
![HgCd Combined](<Plots/Spectra/HgCd Combined.png>)
