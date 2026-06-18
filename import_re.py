import re
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Cálculo de coeficientes KT para TRANSOPTR
# ============================================================

def kt_coefficients_from_a_m_k(a_cm, m, k_cm_inv):
    """
    Calcula A01 y A10 usando el potencial de dos términos.

    a_cm      : apertura mínima en cm
    m         : modulación
    k_cm_inv  : k en cm^-1

    A01 queda en cm^-2
    A10 es adimensional
    """
    ka = k_cm_inv * a_cm
    kma = k_cm_inv * m * a_cm

    I0_ka = np.i0(ka)
    I0_kma = np.i0(kma)

    denom = m**2 * I0_ka + I0_kma

    A10 = (m**2 - 1.0) / denom
    A01 = (1.0 - A10 * I0_ka) / (a_cm**2)

    return A01, A10


# ============================================================
# Lectura robusta de tabla estilo RFQ
# ============================================================

def read_cell_table(filename):
    """
    Lee una tabla con columnas tipo:

    Cell V Wsyn Sig0T Sig0L A10 Phi a m B L Z A0 RFdef Oct A1

    Acepta celdas tipo 1R, 2R, etc.
    Ignora líneas que no empiecen con número/celda.
    """
    rows = []

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            clean = line.strip()

            if not clean:
                continue

            # Ignorar encabezados
            if clean.startswith("#"):
                continue
            if clean.lower().startswith("cell"):
                continue
            if clean.lower().startswith("the column"):
                continue

            parts = clean.split()

            # Debe empezar con algo tipo 0, 1R, 23, 45R
            if not re.match(r"^\d+[A-Za-z]*$", parts[0]):
                continue

            # Algunas líneas pueden estar incompletas, por ejemplo cell 0
            # Tomamos lo que exista.
            cell_label = parts[0]

            try:
                nums = [float(x) for x in parts[1:]]
            except ValueError:
                continue

            rows.append([cell_label] + nums)

    if not rows:
        raise ValueError("No se encontraron filas numéricas en la tabla.")

    return rows


def build_dataframe_from_rows(rows):
    """
    Construye DataFrame. Usa las columnas principales hasta Z.
    """
    base_cols = [
        "Cell", "V_kV", "Wsyn_MeV", "Sig0T_deg", "Sig0L_deg",
        "A10_table", "Phi_deg", "a_cm", "m", "B",
        "L_cm", "Z_cm", "A0", "RFdef", "Oct", "A1"
    ]

    max_len = max(len(r) for r in rows)
    cols = base_cols[:max_len]

    df = pd.DataFrame(rows, columns=cols)

    # Convertir etiqueta de celda a número entero
    df["cell_number"] = df["Cell"].astype(str).str.extract(r"(\d+)").astype(int)

    return df


# ============================================================
# Generación de fort.75 usando Z, a y m
# ============================================================

def generate_fort75_from_cell_table(
    input_table,
    output_fort75="fort.75",
    output_csv="rfq_cell_by_cell_potentials.csv",
    skip_cell0=True,
    a_scale=1.0,
):
    """
    Genera fort.75 desde una tabla RFQ.

    Usa:
        Z_cm como posición acumulada al final de celda
        a_cm como apertura
        m como modulación

    k se calcula celda por celda:
        k = pi / (Z_i - Z_{i-1})

    Parámetros:
    ----------
    input_table : str
        Archivo de entrada con tabla RFQ.
    output_fort75 : str
        Archivo de salida para TRANSOPTR.
    output_csv : str
        Archivo csv de revisión.
    skip_cell0 : bool
        Si True, ignora la celda 0 si no tiene L/Z completos.
    a_scale : float
        Factor para convertir a.
        Usa a_scale=1.0 si la tabla ya está en cm.
        Usa a_scale=0.1 si la tabla está en mm y quieres convertir a cm.
    """

    rows = read_cell_table(input_table)
    df = build_dataframe_from_rows(rows)

    required = {"Z_cm", "a_cm", "m"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Quitar filas sin Z, a o m
    df = df.dropna(subset=["Z_cm", "a_cm", "m"]).copy()

    if skip_cell0:
        df = df[df["cell_number"] > 0].copy()

    df = df.sort_values("cell_number").reset_index(drop=True)

    # Reescalar a si fuera necesario
    df["a_used_cm"] = df["a_cm"].astype(float) * a_scale
    df["m_used"] = df["m"].astype(float)
    df["Z_used_cm"] = df["Z_cm"].astype(float)

    # Hacer que fort.75 empiece en s=0
    # Z es distancia al final de celda desde la entrada.
    # La primera celda empieza en Z=0 y termina en Z_1.
    z_end = df["Z_used_cm"].to_numpy()
    z_start = np.zeros_like(z_end)
    z_start[0] = 0.0
    z_start[1:] = z_end[:-1]

    L_cell = z_end - z_start

    if np.any(L_cell <= 0):
        raise ValueError("Hay celdas con longitud <= 0. Revisa la columna Z.")

    k_cell = np.pi / L_cell

    A01_list = []
    A10_list = []

    for a_cm, m, k in zip(df["a_used_cm"], df["m_used"], k_cell):
        A01, A10 = kt_coefficients_from_a_m_k(a_cm, m, k)
        A01_list.append(A01)
        A10_list.append(A10)

    df["s_start_cm"] = z_start
    df["s_end_cm"] = z_end
    df["L_from_Z_cm"] = L_cell
    df["k_cm_inv"] = k_cell
    df["A01_calc"] = A01_list
    df["A10_calc"] = A10_list

    # Escribir fort.75.
    # Para que TRANSOPTR tenga definido el inicio, escribimos un punto en s=0
    # con los coeficientes de la primera celda.
    with open(output_fort75, "w") as f:
        first = df.iloc[0]
        f.write(
            f"{0.0:18.10E} "
            f"{first['A01_calc']:18.10E} "
            f"{first['A10_calc']:18.10E} "
            f"{first['k_cm_inv']:18.10E}"
            f"   ! cell={first['Cell']} start, a_cm={first['a_used_cm']:.8f}, m={first['m_used']:.8f}\n"
        )

        for _, row in df.iterrows():
            f.write(
                f"{row['s_end_cm']:18.10E} "
                f"{row['A01_calc']:18.10E} "
                f"{row['A10_calc']:18.10E} "
                f"{row['k_cm_inv']:18.10E}"
                f"   ! cell={row['Cell']}, L_cm={row['L_from_Z_cm']:.8f}, "
                f"a_cm={row['a_used_cm']:.8f}, m={row['m_used']:.8f}\n"
            )

    df.to_csv(output_csv, index=False)

    print(f"Archivo TRANSOPTR generado: {output_fort75}")
    print(f"Archivo de revisión generado: {output_csv}")
    print()
    print(f"Número de celdas usadas: {len(df)}")
    print(f"s inicial = 0.0 cm")
    print(f"s final   = {df['s_end_cm'].iloc[-1]:.10E} cm")
    print()
    print("Primeras celdas:")
    print(df[["Cell", "s_start_cm", "s_end_cm", "L_from_Z_cm", "a_used_cm", "m_used", "A01_calc", "A10_calc", "k_cm_inv"]].head())
    print()
    print("Últimas celdas:")
    print(df[["Cell", "s_start_cm", "s_end_cm", "L_from_Z_cm", "a_used_cm", "m_used", "A01_calc", "A10_calc", "k_cm_inv"]].tail())

    return df


# ============================================================
# Ejecución directa
# ============================================================

if __name__ == "__main__":
    df = generate_fort75_from_cell_table(
        input_table="table2_dans.txt",
        output_fort75="fort.75",
        output_csv="rfq_cell_by_cell_potentials.csv",
        skip_cell0=True,

        # Si la columna a está realmente en cm, deja 1.0.
        # Si descubres que está en mm, cambia a 0.1.
        a_scale=1.0,
    )
