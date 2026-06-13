#!/usr/bin/env python3
"""Grafica espacios fase leyendo solamente cross_z0p300.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path(__file__).with_name("cross_z0p300.csv")
OUTPUT_PATH = Path(__file__).with_name("phase_space_cross_z0p300.png")

RF_FREQUENCY_HZ = 35.435e6
RF_FREQUENCY_HZ = 162.0e6
REST_ENERGY_MEV = 938.2720813  # H- aproximado por la masa del proton
LIGHT_SPEED = 299_792_458.0


def rms_emittance(position_m, angle_rad):
    """Calcula la emitancia geometrica RMS en m rad."""
    position = np.asarray(position_m, dtype=float)
    angle = np.asarray(angle_rad, dtype=float)

    position = position - np.mean(position)
    angle = angle - np.mean(angle)

    determinant = (
        np.mean(position**2) * np.mean(angle**2)
        - np.mean(position * angle) ** 2
    )
    return np.sqrt(max(determinant, 0.0))


def kinetic_energy_mev(vx, vy, vz):
    """Calcula la energia cinetica relativista en MeV."""
    speed_squared = vx**2 + vy**2 + vz**2
    beta_squared = np.clip(
        speed_squared / LIGHT_SPEED**2,
        0.0,
        1.0 - 1e-15,
    )
    gamma = 1.0 / np.sqrt(1.0 - beta_squared)
    return (gamma - 1.0) * REST_ENERGY_MEV
def kinetic_energy_mev_nonrel(vx, vy, vz):
    """Calcula la energia cinetica no relativista en MeV."""
    speed_squared = vx**2 + vy**2 + vz**2
    return 0.5 * speed_squared * REST_ENERGY_MEV / LIGHT_SPEED**2


def rf_phase_deg(time_s):
    """Convierte el tiempo de cruce a fase RF entre -180 y 180 grados."""
    phase = 360.0 * RF_FREQUENCY_HZ * (time_s - np.mean(time_s))
    return (phase + 180.0) % 360.0 - 180.0
def time_to_phase_deg(t, f_rf, tref=0.0, phi0_deg=0.0, wrap=True, center180=False):
    """
    t: array-like de tiempos [s]
    f_rf: frecuencia [Hz]
    tref: tiempo de referencia [s] (se usa t - tref)
    phi0_deg: offset de fase [deg]
    wrap: si True, aplica módulo 360
    center180: si True, devuelve en (-180, 180]; requiere wrap=True
    """
    phi = 360.0 * f_rf * (np.asarray(t) - tref) + phi0_deg
    if wrap:
        phi = np.mod(phi, 360.0)
        if center180:
            phi = np.where(phi > 180.0, phi - 360.0, phi)
    return phi


def density_plot(axis, x, y, xlabel, ylabel, title, bins=180):
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    visible_counts = np.ma.masked_equal(counts.T, 0)
    image = axis.pcolormesh(
        x_edges,
        y_edges,
        visible_counts,
        cmap="turbo",
        shading="auto",
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.2)
    plt.colorbar(image, ax=axis, label="Particulas")


def main():
    required_columns = ["t_cross", "x", "y", "vx", "vy", "vz"]
    data = pd.read_csv(CSV_PATH)

    missing = [column for column in required_columns if column not in data]
    if missing:
        raise ValueError(f"Faltan columnas en {CSV_PATH.name}: {missing}")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required_columns)
    data = data[data["vz"] != 0.0]

    x_mm = data["x"].to_numpy() * 1e3
    y_mm = data["y"].to_numpy() * 1e3
    xp_mrad = data["vx"].to_numpy() / data["vz"].to_numpy() * 1e3
    yp_mrad = data["vy"].to_numpy() / data["vz"].to_numpy() * 1e3

    energy_mev = kinetic_energy_mev_nonrel(
        data["vx"].to_numpy(),
        data["vy"].to_numpy(),
        data["vz"].to_numpy(),
    )
    phase_deg = rf_phase_deg(data["t_cross"].to_numpy())
    phase_deg = time_to_phase_deg(
        data["t_cross"].to_numpy(),
        f_rf=RF_FREQUENCY_HZ,
        tref=data["t_cross"].to_numpy().min(),
        phi0_deg=90.0,
        wrap=True,
        center180=False,
    )
    #phitime= time_to_phase_deg(m["t2"], f_rf=freq_rfq_0, tref=m["t2"].min(), phi0_deg=90.0, wrap=True, center180=False)


    emit_x = rms_emittance(x_mm * 1e-3, xp_mrad * 1e-3)
    emit_y = rms_emittance(y_mm * 1e-3, yp_mrad * 1e-3)

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))

    density_plot(
        axes[0, 0],
        x_mm,
        xp_mrad,
        "x [mm]",
        "x' [mrad]",
        f"Espacio fase X\nemitancia RMS = {emit_x * 1e6:.4f} mm mrad",
    )
    density_plot(
        axes[0, 1],
        y_mm,
        yp_mrad,
        "y [mm]",
        "y' [mrad]",
        f"Espacio fase Y\nemitancia RMS = {emit_y * 1e6:.4f} mm mrad",
    )
    density_plot(
        axes[1, 0],
        x_mm,
        y_mm,
        "x [mm]",
        "y [mm]",
        "Distribucion transversal",
    )
    density_plot(
        axes[1, 1],
        phase_deg,
        energy_mev,
        "Fase RF [grados]",
        "Energia cinetica [MeV]",
        "Espacio fase longitudinal",
    )

    figure.suptitle(
        f"{CSV_PATH.name}: {len(data)} particulas",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=180)

    print(f"Particulas leidas: {len(data)}")
    print(f"Emitancia RMS X: {emit_x * 1e6:.6g} mm mrad")
    print(f"Emitancia RMS Y: {emit_y * 1e6:.6g} mm mrad")
    print(f"Energia media: {np.mean(energy_mev):.6g} MeV")
    print(f"Grafica guardada en: {OUTPUT_PATH}")

    plt.figure(figsize=(8, 5))
    plt.hist(phase_deg, bins=180, alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()
