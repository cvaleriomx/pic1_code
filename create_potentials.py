#Tabla semilla de RFQ para protones
#Objetivo: 30 keV -> 2.0 MeV
#Frecuencia: 714 MHz


#Columnas:
#cell       = número de celda
#section    = sección asignada en la tabla semilla
#Win_keV    = energía cinética de entrada de la celda
#Wout_keV   = energía cinética de salida de la celda
#a_mm       = apertura mínima adoptada en la celda
#m          = modulación adoptada en la celda
#phi_s_deg  = fase sincrónica adoptada en la celda
#U_kV       = voltaje intervane adoptado en la celda
#Lc_mm      = longitud de celda

#Sobre cómo se calculó la energía del haz y U en esta tabla:
#1) U NO es la energía del haz. U es el voltaje intervane del RFQ.
#2) La energía del haz está en Win_keV y Wout_keV.
#3) En esta tabla semilla, Win_keV y Wout_keV NO se obtuvieron integrando todavía
#   el campo longitudinal real del RFQ. Se repartieron de forma heurística por secciones:
#   - RM: sin aceleración (30 -> 30 keV)
#   - SH: incremento muy pequeño
#   - GB: bunching con ganancia pequeña
#   - ACC-1 y ACC-2: ganancia principal de energía hasta 2 MeV
#4) U_kV también se definió de forma programada y suave por secciones:
#   - 40 kV al inicio
#   - aumento gradual hasta 50 kV al final
#5) Lc_mm se tomó como longitud de celda RFQ:
#      Lc = beta * lambda / 2
#   usando beta calculada con la energía media de la celda:
#      Wmid = (Win_keV + Wout_keV)/2
#   para un protón y lambda = c/f con f = 714 MHz.
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Constantes físicas
# ============================================================
C = 299_792_458.0                  # m/s
PROTON_REST_ENERGY_KEV = 938_272.0813
DEFAULT_FREQUENCY_HZ = 714e6


# ============================================================
# Cinemática relativista
# ============================================================
def gamma_from_kinetic_energy_keV(
    w_keV: float,
    rest_keV: float = PROTON_REST_ENERGY_KEV
) -> float:
    return 1.0 + w_keV / rest_keV


def beta_from_kinetic_energy_keV(
    w_keV: float,
    rest_keV: float = PROTON_REST_ENERGY_KEV
) -> float:
    gamma = gamma_from_kinetic_energy_keV(w_keV, rest_keV)
    return math.sqrt(1.0 - 1.0 / (gamma * gamma))


# ============================================================
# Lectura de la tabla RFQ
# Se espera un archivo con SOLO esta tabla:
# cell section Win_keV Wout_keV a_mm m phi_s_deg U_kV Lc_mm
# ============================================================
def read_rfq_text_table(txt_path: str | Path) -> pd.DataFrame:
    txt_path = Path(txt_path)
    return pd.read_csv(txt_path, sep=r"\s+", engine="python")


# ============================================================
# Coeficientes según el paper de TRANSOPTR RFQ
#
# A10 = (m^2 - 1) / (m^2 I0(ka) + I0(mka))
# A01 = (1 - A10 I0(ka)) / a^2
#
# Aquí a debe estar en cm si k está en cm^-1
# Entonces A01 queda en cm^-2
# ============================================================
def compute_a_coefficients_transoptr(a_cm: float, m: float, k_per_cm: float) -> tuple[float, float]:
    ka = k_per_cm * a_cm
    mka = m * ka

    I0_ka = np.i0(ka)
    I0_mka = np.i0(mka)

    A10 = ((m ** 2) - 1.0) / ((m ** 2) * I0_ka + I0_mka)
    A01 = (1.0 - A10 * I0_ka) / (a_cm ** 2)

    return float(A01), float(A10)


# ============================================================
# Puntos por celda
# Ajusta esta función como quieras
# ============================================================
def points_per_cell(section: str) -> int:
    section = str(section).strip().upper()

    if section in {"RM", "SH"}:
        return 5
    if section in {"GB", "MS", "MB"}:
        return 3
    if section in {"ACC-1", "ACC-2", "MBA", "MA"}:
        return 2
    return 2


# ============================================================
# Generador principal
#
# Salida:
#   fort.755 con 4 columnas numéricas:
#   s[cm]  A01[cm^-2]  A10[-]  k[cm^-1]
# ============================================================
def build_transoptr_potential_file(
    input_txt: str | Path,
    output_fort: str | Path = "fort.755",
    output_csv: str | Path = "rfq_with_potentials.csv",
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    use_lc_for_k: bool = True,
    include_final_endpoint: bool = True,
) -> pd.DataFrame:

    df = read_rfq_text_table(input_txt)

    required_cols = {"cell", "section", "Win_keV", "Wout_keV", "a_mm", "m", "Lc_mm"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    wavelength_m = C / frequency_hz
    wavelength_cm = wavelength_m * 100.0

    rows = []
    s_running_cm = 0.0

    for _, row in df.iterrows():
        cell = int(row["cell"])
        section = str(row["section"])
        win = float(row["Win_keV"])
        wout = float(row["Wout_keV"])
        a_cm = float(row["a_mm"]) / 10.0
        m = float(row["m"])
        lc_cm = float(row["Lc_mm"]) / 10.0

        npts = points_per_cell(section)
        if npts < 1:
            npts = 1

        # puntos dentro de la celda, incluyendo el inicio y excluyendo el final
        # para evitar duplicar el punto con la siguiente celda
        frac_positions = np.linspace(0.0, 1.0, npts, endpoint=False)

        for frac in frac_positions:
            # interpolación lineal local de energía
            w_loc = win + frac * (wout - win)

            if use_lc_for_k:
                # consistente con la tabla: Lc = pi / k
                k_per_cm = math.pi / lc_cm
            else:
                beta = beta_from_kinetic_energy_keV(w_loc)
                k_per_cm = 2.0 * math.pi / (beta * wavelength_cm)

            A01, A10 = compute_a_coefficients_transoptr(
                a_cm=a_cm,
                m=m,
                k_per_cm=k_per_cm,
            )

            s_loc_cm = s_running_cm + frac * lc_cm

            rows.append({
                "cell": cell,
                "section": section,
                "s_cm": s_loc_cm,
                "W_keV_local": w_loc,
                "a_cm": a_cm,
                "m": m,
                "A01_cm_inv2": A01,
                "A10": A10,
                "k_cm_inv": k_per_cm,
            })

        s_running_cm += lc_cm

    # punto final opcional
    if include_final_endpoint and len(df) > 0:
        last = df.iloc[-1]
        last_cell = int(last["cell"])
        last_section = str(last["section"])
        last_a_cm = float(last["a_mm"]) / 10.0
        last_m = float(last["m"])
        last_w = float(last["Wout_keV"])
        last_lc_cm = float(last["Lc_mm"]) / 10.0

        if use_lc_for_k:
            k_per_cm = math.pi / last_lc_cm
        else:
            beta = beta_from_kinetic_energy_keV(last_w)
            k_per_cm = 2.0 * math.pi / (beta * wavelength_cm)

        A01, A10 = compute_a_coefficients_transoptr(
            a_cm=last_a_cm,
            m=last_m,
            k_per_cm=k_per_cm,
        )

        rows.append({
            "cell": last_cell,
            "section": last_section,
            "s_cm": s_running_cm,
            "W_keV_local": last_w,
            "a_cm": last_a_cm,
            "m": last_m,
            "A01_cm_inv2": A01,
            "A10": A10,
            "k_cm_inv": k_per_cm,
        })

    out_df = pd.DataFrame(rows)

    # escribir fort.755 sin encabezados
    output_fort = Path(output_fort)
    with output_fort.open("w", encoding="utf-8") as f:
        for _, row in out_df.iterrows():
            f.write(
                f"{row['s_cm']: .10E}  "
                f"{row['A01_cm_inv2']: .10E}  "
                f"{row['A10']: .10E}  "
                f"{row['k_cm_inv']: .10E}\n"
            )

    output_csv = Path(output_csv)
    out_df.to_csv(output_csv, index=False)

    return out_df


# ============================================================
# Principal
# ============================================================
if __name__ == "__main__":
    out = build_transoptr_potential_file(
        input_txt="rfq_table_180cells_714MHz_50keV_4MeV_with_Lc.txt",
        output_fort="fort.755",
        output_csv="rfq_with_potentials.csv",
        frequency_hz=714e6,
        use_lc_for_k=True,
        include_final_endpoint=True,
    )

    print("Archivo generado: fort.755")
    print("Tabla enriquecida generada: rfq_with_potentials.csv")
    print(f"Número de puntos generados: {len(out)}")
    print(f"Primer s = {out['s_cm'].iloc[0]:.10E} cm")
    print(f"Último s = {out['s_cm'].iloc[-1]:.10E} cm")
    print("")
    print("Usa en sy.f:")
    print(f"call rfq(755,{len(out)},vane,760.1206,714.0E+06,rfp,nscav)")
