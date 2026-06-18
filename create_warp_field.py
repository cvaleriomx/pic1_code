import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuración principal
# ============================================================

INPUT_TABLE = "table1.txt"
INPUT_POTENTIALS_TABLE = "ISAC.rfq_potentials.txt"
INPUT_POTENTIALS_TABLE = "RM2/olivier_1/fort.76"
OUTPUT_PKL = "campo_corregido_ISAC2.pkl"
PLOT_DIR = Path("fieldmap_diagnostics")
PLOT_DIR.mkdir(exist_ok=True)

# "geometry": usa INPUT_TABLE y toma A10 desde la tabla RFQ.
# "geometry_kt": usa INPUT_TABLE, pero calcula A01/A10 desde a, m y k.
# "potentials": usa INPUT_POTENTIALS_TABLE con A01/A10/k ya calculados.
# "potentials_uniform": interpola esos potenciales a una malla z uniforme.
COEFFICIENT_SOURCE = "potentials_uniform"

# Frecuencia RF
FREQ_HZ = 162e6

# ------------------------------------------------------------
# Malla en cm
# Tú defines explícitamente límites y pasos.
# ------------------------------------------------------------

x_min_cm = -1.0
x_max_cm =  1.0
dx_cm = 0.02

y_min_cm = -1.0
y_max_cm =  1.0
dy_cm = 0.02

z_min_cm = 0.0
dz_cm = 0.05

# Paso usado para remuestrear potenciales no uniformes cuando se usa
# COEFFICIENT_SOURCE = "potentials_uniform".
POTENTIALS_UNIFORM_DZ_CM = dz_cm

# Si z_max_cm = None, se toma del último Z de la tabla.
z_max_cm = None

# ------------------------------------------------------------
# Términos del potencial
# ------------------------------------------------------------

USE_A0 = True
USE_A10 = True

# Activar después de validar A0 y A10.
USE_OCT = False
USE_A1 = False

# Si ya tienes una tabla de potenciales por celda, puedes usar:
#     generate_field_map_from_potentials()
#
# Formatos aceptados:
#   CSV con columnas tipo s_cm/Z_cm/s_end_cm, A01_calc/A10_calc/k_cm_inv
#   CSV con columnas tipo s_cm, A01_fit/A10_fit/A03_fit/A05_fit/k_cm_inv
#   archivo tipo fort.75: s_cm A01 A10 k_cm_inv [A03 A05]


# ============================================================
# Lectura de tabla RFQ
# ============================================================

def read_rfq_table(filename):
    rows = []

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            if not re.match(r"^\d+[A-Za-z]*$", parts[0]):
                continue

            try:
                nums = [float(x) for x in parts[1:]]
            except ValueError:
                continue

            rows.append([parts[0]] + nums)

    if not rows:
        raise ValueError("No encontré filas numéricas en la tabla RFQ.")

    cols = [
        "Cell", "V_kV", "Wsyn_MeV", "Sig0T_deg", "Sig0L_deg",
        "A10", "Phi_deg", "a_cm", "m", "B", "L_cm", "Z_cm",
        "A0", "RFdef", "Oct", "A1"
    ]

    max_len = max(len(r) for r in rows)
    df = pd.DataFrame(rows, columns=cols[:max_len])

    df["cell_number"] = (
        df["Cell"].astype(str)
        .str.extract(r"(\d+)")
        .astype(int)
    )

    df = df.sort_values("cell_number").reset_index(drop=True)

    return df


# ============================================================
# Construcción de ejes con paso fijo
# ============================================================

def make_axis(min_val, max_val, step):
    """
    Construye un eje uniforme usando el paso deseado.
    Ajusta el número de puntos para evitar errores de arange.
    """
    n = int(round((max_val - min_val) / step)) + 1
    axis = min_val + step * np.arange(n)

    # Forzar último punto exactamente al valor esperado por consistencia.
    axis[-1] = min_val + step * (n - 1)

    return axis


# ============================================================
# Preparar coeficientes
# ============================================================

def kt_coefficients_from_a_m_k(a_cm, m, k_cm_inv):
    """
    Calcula A01 y A10 desde apertura, modulación y k.

    Es la misma fórmula usada en import_re.py:

        A10 = (m^2 - 1) / (m^2 I0(k a) + I0(k m a))
        A01 = (1 - A10 I0(k a)) / a^2

    a_cm y k_cm_inv deben estar en unidades consistentes de cm.
    """
    ka = k_cm_inv * a_cm
    kma = k_cm_inv * m * a_cm

    I0_ka = np.i0(ka)
    I0_kma = np.i0(kma)

    denom = m**2 * I0_ka + I0_kma

    A10 = (m**2 - 1.0) / denom
    A01 = (1.0 - A10 * I0_ka) / (a_cm**2)

    return A01, A10


def prepare_coefficients(df):
    """
    De la tabla construye funciones tabuladas:

        s = Z
        k = pi / L
        A01 = A0 / a^2
        A10 = A10_tabla

    y coeficientes aproximados para términos superiores:

        A03 = Oct / a^4
        A05 = A1 / a^6

    La fase espacial se calcula como:

        psi(z) = integral k dz
    """

    required = ["Z_cm", "L_cm", "a_cm", "A10"]
    df = df.dropna(subset=required).copy()

    df = df[df["Z_cm"] > 0].copy()
    df = df[df["L_cm"] > 0].copy()

    s = df["Z_cm"].to_numpy()
    L = df["L_cm"].to_numpy()
    a = df["a_cm"].to_numpy()

    k = np.pi / L

    A10 = df["A10"].to_numpy()

    if "A0" in df.columns:
        A0 = df["A0"].fillna(1.0).to_numpy()
    else:
        A0 = np.ones_like(a)

    A01 = A0 / a**2

    if "Oct" in df.columns:
        Oct = df["Oct"].fillna(0.0).to_numpy()
    else:
        Oct = np.zeros_like(a)

    if "A1" in df.columns:
        A1 = df["A1"].fillna(0.0).to_numpy()
    else:
        A1 = np.zeros_like(a)

    A03 = Oct / a**4
    A05 = A1 / a**6

    # Construir psi acumulada por celda
    s_start = np.zeros_like(s)
    s_start[0] = 0.0
    s_start[1:] = s[:-1]

    ds = s - s_start
    psi_end = np.cumsum(k * ds)

    # Agregar punto inicial s=0
    s_ext = np.concatenate([[0.0], s])
    psi_ext = np.concatenate([[0.0], psi_end])

    coeff = {
        "s": s_ext,
        "psi": psi_ext,
        "k": np.concatenate([[k[0]], k]),
        "A01": np.concatenate([[A01[0]], A01]),
        "A10": np.concatenate([[A10[0]], A10]),
        "A03": np.concatenate([[A03[0]], A03]),
        "A05": np.concatenate([[A05[0]], A05]),
        "a": np.concatenate([[a[0]], a]),
        "z_max": s[-1],
    }

    return coeff


def prepare_coefficients_from_geometry_kt(df):
    """
    Construye coeficientes desde geometría, sin usar A10 de la tabla.

    Usa:
        s = Z_cm
        k = pi / L_cm
        A01, A10 = kt_coefficients_from_a_m_k(a_cm, m, k)

    Esto es útil cuando la tabla tiene apertura y modulación confiables, pero
    no quieres usar el potencial A10 tabulado.
    """

    required = ["Z_cm", "L_cm", "a_cm", "m"]
    df = df.dropna(subset=required).copy()

    df = df[df["Z_cm"] > 0].copy()
    df = df[df["L_cm"] > 0].copy()
    df = df[df["a_cm"] > 0].copy()

    if df.empty:
        raise ValueError("No quedaron filas válidas con Z_cm, L_cm, a_cm y m.")

    s = df["Z_cm"].to_numpy()
    L = df["L_cm"].to_numpy()
    a = df["a_cm"].to_numpy()
    m = df["m"].to_numpy()

    k = np.pi / L
    A01, A10 = kt_coefficients_from_a_m_k(a, m, k)

    if "Oct" in df.columns:
        Oct = df["Oct"].fillna(0.0).to_numpy()
    else:
        Oct = np.zeros_like(a)

    if "A1" in df.columns:
        A1 = df["A1"].fillna(0.0).to_numpy()
    else:
        A1 = np.zeros_like(a)

    A03 = Oct / a**4
    A05 = A1 / a**6

    s_start = np.zeros_like(s)
    s_start[0] = 0.0
    s_start[1:] = s[:-1]

    ds = s - s_start
    psi_end = np.cumsum(k * ds)

    s_ext = np.concatenate([[0.0], s])
    psi_ext = np.concatenate([[0.0], psi_end])

    coeff = {
        "s": s_ext,
        "psi": psi_ext,
        "k": np.concatenate([[k[0]], k]),
        "A01": np.concatenate([[A01[0]], A01]),
        "A10": np.concatenate([[A10[0]], A10]),
        "A03": np.concatenate([[A03[0]], A03]),
        "A05": np.concatenate([[A05[0]], A05]),
        "a": np.concatenate([[a[0]], a]),
        "z_max": s[-1],
    }

    return coeff


# ============================================================
# Preparar coeficientes desde potenciales ya calculados
# ============================================================

def read_potential_coefficients_table(filename):
    """
    Lee una lista de potenciales/coefs ya calculados por celda.

    Acepta CSV con encabezado, o archivos tipo fort.75 con columnas:

        s_cm  A01  A10  k_cm_inv  [A03]  [A05]

    En este camino no se requiere apertura a ni modulación m.
    """
    path = Path(filename)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.split("!", 1)[0]
            line = line.split("#", 1)[0].strip()

            if not line:
                continue

            parts = line.replace(",", " ").split()
            values = []

            for part in parts:
                try:
                    values.append(float(part))
                except ValueError:
                    break

            if len(values) >= 4:
                rows.append(values[:6])

    if not rows:
        raise ValueError(f"No encontré filas numéricas de potenciales en {filename}.")

    max_len = max(len(row) for row in rows)
    cols = ["s_cm", "A01", "A10", "k_cm_inv", "A03", "A05"][:max_len]

    return pd.DataFrame(rows, columns=cols)


def _first_existing_column(df, candidates):
    lower_to_col = {str(col).lower(): col for col in df.columns}

    for candidate in candidates:
        col = lower_to_col.get(candidate.lower())
        if col is not None:
            return col

    return None


def _required_numeric_column(df, candidates, label):
    col = _first_existing_column(df, candidates)

    if col is None:
        raise ValueError(
            f"Falta columna para {label}. Probé estos nombres: {', '.join(candidates)}"
        )

    return pd.to_numeric(df[col], errors="coerce")


def _optional_numeric_column(df, candidates, default_value=0.0):
    col = _first_existing_column(df, candidates)

    if col is None:
        return pd.Series(default_value, index=df.index, dtype=float)

    return pd.to_numeric(df[col], errors="coerce").fillna(default_value)


def _phase_from_tabulated_k(s, k):
    """
    Integra psi = integral k dz usando el k del punto derecho.

    Esto conserva la convención que ya usaba prepare_coefficients(): cada
    fila de fin de celda define el k de la celda que termina en ese s.
    """
    psi = np.zeros_like(s, dtype=float)

    if len(s) > 1:
        ds = np.diff(s)
        psi[1:] = np.cumsum(k[1:] * ds)

    return psi


def prepare_coefficients_from_potentials(table_or_df):
    """
    Construye el diccionario coeff desde potenciales ya calculados.

    Columnas reconocidas:
        posición: s_cm, Z_cm, Z_used_cm, s_end_cm, z_cm, s, Z
        A01:      A01, A01_calc, A01_fit, A01_cm_inv2
        A10:      A10, A10_calc, A10_fit, A10_table
        k:        k_cm_inv, k, K

    Si no existe k, se calcula con L:
        k = pi / L_cm
    usando L_cm, L_from_Z_cm o L_cm_used.

    A03 y A05 son opcionales y se leen solo si vienen ya como coeficientes
    del potencial, por ejemplo A03_fit/A05_fit.
    """
    if isinstance(table_or_df, (str, Path)):
        df = read_potential_coefficients_table(table_or_df)
    else:
        df = table_or_df.copy()

    s = _required_numeric_column(
        df,
        ["s_cm", "Z_cm", "Z_used_cm", "s_end_cm", "z_cm", "s", "Z"],
        "posición longitudinal s/Z [cm]",
    )

    A01 = _required_numeric_column(
        df,
        ["A01", "A01_calc", "A01_fit", "A01_cm_inv2"],
        "A01",
    )
    A10 = _required_numeric_column(
        df,
        ["A10", "A10_calc", "A10_fit", "A10_table"],
        "A10",
    )

    k_col = _first_existing_column(df, ["k_cm_inv", "k", "K"])
    if k_col is None:
        L = _required_numeric_column(
            df,
            ["L_cm", "L_from_Z_cm", "L_cm_used", "Lc_cm"],
            "k_cm_inv o longitud L [cm]",
        )
        k = np.pi / L
    else:
        k = pd.to_numeric(df[k_col], errors="coerce")

    A03 = _optional_numeric_column(
        df,
        ["A03", "A03_calc", "A03_fit", "A03_cm_inv4"],
    )
    A05 = _optional_numeric_column(
        df,
        ["A05", "A05_calc", "A05_fit", "A05_cm_inv6"],
    )

    work = pd.DataFrame({
        "s": s,
        "k": k,
        "A01": A01,
        "A10": A10,
        "A03": A03,
        "A05": A05,
    })

    psi_col = _first_existing_column(df, ["psi", "psi_rad", "phase_rad"])
    if psi_col is not None:
        work["psi"] = pd.to_numeric(df[psi_col], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan)
    required = ["s", "k", "A01", "A10"]
    if "psi" in work.columns:
        required.append("psi")

    work = work.dropna(subset=required).sort_values("s")
    work = work.drop_duplicates(subset="s", keep="last").reset_index(drop=True)

    if work.empty:
        raise ValueError("La tabla de potenciales no dejó filas válidas.")

    if np.any(np.diff(work["s"].to_numpy()) <= 0):
        raise ValueError("La coordenada s/Z debe ser estrictamente creciente.")

    if work["s"].iloc[0] > 0.0:
        first = work.iloc[0].copy()
        first["s"] = 0.0
        if "psi" in work.columns:
            first["psi"] = 0.0
        work = pd.concat([first.to_frame().T, work], ignore_index=True)

    s_array = work["s"].to_numpy(dtype=float)
    k_array = work["k"].to_numpy(dtype=float)

    if "psi" in work.columns:
        psi = work["psi"].to_numpy(dtype=float)
    else:
        psi = _phase_from_tabulated_k(s_array, k_array)

    coeff = {
        "s": s_array,
        "psi": psi,
        "k": k_array,
        "A01": work["A01"].to_numpy(dtype=float),
        "A10": work["A10"].to_numpy(dtype=float),
        "A03": work["A03"].to_numpy(dtype=float),
        "A05": work["A05"].to_numpy(dtype=float),
        "a": np.full_like(s_array, np.nan, dtype=float),
        "z_max": s_array[-1],
    }

    return coeff


def resample_coefficients_to_uniform_z(coeff, dz=None, nz=None, z_max=None):
    """
    Remuestrea coeficientes tabulados en s no uniforme a z uniforme.

    Equivale a:

        z = np.linspace(0, L, nz)
        A01_z = np.interp(z, s, A01)
        A10_z = np.interp(z, s, A10)
        k_z   = np.interp(z, s, k)

    La fase se interpola desde la fase acumulada original:

        psi_z = interp(z, s, psi)

    Esto preserva psi=n*pi en cada frontera de celda. Integrar de nuevo una
    k interpolada suavemente cambia la longitud de fase de cada celda y
    produce una deriva longitudinal acumulativa.
    """
    s = coeff["s"]

    if z_max is None:
        z_max = coeff["z_max"]

    if dz is None:
        dz = POTENTIALS_UNIFORM_DZ_CM

    if nz is None:
        if dz <= 0:
            raise ValueError("dz debe ser > 0 para remuestrear potenciales.")
        nz = int(round(z_max / dz)) + 1

    if nz < 2:
        raise ValueError("nz debe ser >= 2 para remuestrear potenciales.")

    z = np.linspace(0.0, z_max, nz)

    k_z = np.interp(z, s, coeff["k"])
    psi_z = np.interp(z, s, coeff["psi"])

    uniform = {
        "s": z,
        "psi": psi_z,
        "k": k_z,
        "A01": np.interp(z, s, coeff["A01"]),
        "A10": np.interp(z, s, coeff["A10"]),
        "A03": np.interp(z, s, coeff["A03"]),
        "A05": np.interp(z, s, coeff["A05"]),
        "a": np.full_like(z, np.nan, dtype=float),
        "z_max": z[-1],
    }

    return uniform


def prepare_coefficients_from_potentials_uniform(
    table_or_df,
    dz=None,
    nz=None,
    z_max=None,
):
    """
    Lee potenciales A01/A10/k no uniformes y los remuestrea a z uniforme.

    Usa las mismas columnas/formato que prepare_coefficients_from_potentials(),
    pero antes de calcular el mapa final convierte la tabla a una malla
    longitudinal uniforme.
    """
    coeff = prepare_coefficients_from_potentials(table_or_df)

    return resample_coefficients_to_uniform_z(
        coeff,
        dz=dz,
        nz=nz,
        z_max=z_max,
    )


# ============================================================
# Interpolación de coeficientes
# ============================================================

def interp_coefficients(Z, coeff):
    s = coeff["s"]

    c = {}
    for key in ["psi", "k", "A01", "A10", "A03", "A05", "a"]:
        c[key] = np.interp(Z, s, coeff[key])

    return c


# ============================================================
# Potencial espacial normalizado
# ============================================================

def potential_normalized_per_volt(X, Y, Z, coeff):
    """
    Potencial espacial para V_intervane = 1 V.

    X,Y,Z en cm.

    Phi queda en volts por 1 V de intervane.

    Modelo:
        Phi = 1/2 [
            A01 (x^2-y^2)
          + A10 I0(k r) cos(psi)
          + A03 r^4 cos(4 theta)
          + A05 r^6 cos(6 theta)
        ]

    Campo:
        E = -grad(Phi)
    """

    c = interp_coefficients(Z, coeff)

    psi = c["psi"]
    k = c["k"]
    A01 = c["A01"]
    A10 = c["A10"]
    A03 = c["A03"]
    A05 = c["A05"]

    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    F = np.zeros_like(X)

    if USE_A0:
        F += A01 * (X**2 - Y**2)

    if USE_A10:
        F += A10 * np.i0(k * R) * np.cos(psi)

    if USE_OCT:
        F += A03 * R**4 * np.cos(4.0 * theta)

    if USE_A1:
        F += A05 * R**6 * np.cos(6.0 * theta)

    Phi = 0.5 * F

    return Phi


# ============================================================
# Generar campo
# ============================================================

def generate_field_map_from_coefficients(coeff, output_pkl=OUTPUT_PKL):
    global z_max_cm

    if z_max_cm is None:
        z_max = coeff["z_max"]
    else:
        z_max = z_max_cm

    x = make_axis(x_min_cm, x_max_cm, dx_cm)
    y = make_axis(y_min_cm, y_max_cm, dy_cm)
    z = make_axis(z_min_cm, z_max, dz_cm)

    print("Malla generada:")
    print(f"x: {x[0]:.6f} a {x[-1]:.6f} cm, nx={len(x)}, dx={x[1]-x[0]:.6f} cm")
    print(f"y: {y[0]:.6f} a {y[-1]:.6f} cm, ny={len(y)}, dy={y[1]-y[0]:.6f} cm")
    print(f"z: {z[0]:.6f} a {z[-1]:.6f} cm, nz={len(z)}, dz={z[1]-z[0]:.6f} cm")

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    Phi = potential_normalized_per_volt(X, Y, Z, coeff)

    # Gradiente en V/cm
    dPhi_dx, dPhi_dy, dPhi_dz = np.gradient(
        Phi,
        dx_cm,
        dy_cm,
        dz_cm,
        edge_order=2
    )

    Ex_V_per_cm = -dPhi_dx
    Ey_V_per_cm = -dPhi_dy
    Ez_V_per_cm = -dPhi_dz

    # Convertimos a V/m por volt de intervane
    Ex = 100.0 * Ex_V_per_cm
    Ey = 100.0 * Ey_V_per_cm
    Ez = 100.0 * Ez_V_per_cm

    # Guardar DataFrame compatible con tu lectura.
    # Coordenadas en cm. Campos en V/m por volt de intervane.
    df_out = pd.DataFrame({
        "x": X.ravel(order="C"),
        "y": Y.ravel(order="C"),
        "z": Z.ravel(order="C"),
        "Ex": Ex.ravel(order="C"),
        "Ey": Ey.ravel(order="C"),
        "Ez": Ez.ravel(order="C"),
    })

    df_out = df_out.sort_values(by=["x", "y", "z"]).reset_index(drop=True)

    with open(output_pkl, "wb") as f:
        pickle.dump(df_out, f)

    print()
    print(f"Archivo generado: {output_pkl}")
    print(f"Puntos totales: {len(df_out)}")
    print()
    print("Rangos de campo normalizado por 1 V de intervane:")
    print("Ex [V/m/V]:", df_out["Ex"].min(), df_out["Ex"].max())
    print("Ey [V/m/V]:", df_out["Ey"].min(), df_out["Ey"].max())
    print("Ez [V/m/V]:", df_out["Ez"].min(), df_out["Ez"].max())

    make_diagnostic_plots(x, y, z, Ex, Ey, Ez)

    return df_out


def generate_field_map(input_table=INPUT_TABLE, output_pkl=OUTPUT_PKL):
    """
    Genera el campo desde la tabla RFQ original.

    Este camino usa apertura/modulación/tabla RFQ para preparar los
    coeficientes.
    """
    table = read_rfq_table(input_table)
    coeff = prepare_coefficients(table)

    return generate_field_map_from_coefficients(coeff, output_pkl=output_pkl)


def generate_field_map_from_geometry_kt(input_table=INPUT_TABLE, output_pkl=OUTPUT_PKL):
    """
    Genera el campo desde geometría, calculando A01/A10 con a, m y k.

    A diferencia de generate_field_map(), esta función no usa la columna A10
    del archivo de entrada.
    """
    table = read_rfq_table(input_table)
    coeff = prepare_coefficients_from_geometry_kt(table)

    return generate_field_map_from_coefficients(coeff, output_pkl=output_pkl)


def generate_field_map_from_potentials(
    input_table=INPUT_POTENTIALS_TABLE,
    output_pkl=OUTPUT_PKL,
):
    """
    Genera el campo desde una tabla que ya contiene potenciales/coefs.

    No usa apertura a ni modulación m; solo lee A01, A10, k y opcionalmente
    A03/A05.
    """
    coeff = prepare_coefficients_from_potentials(input_table)

    return generate_field_map_from_coefficients(coeff, output_pkl=output_pkl)


def generate_field_map_from_potentials_uniform(
    input_table=INPUT_POTENTIALS_TABLE,
    output_pkl=OUTPUT_PKL,
    dz=None,
    nz=None,
):
    """
    Genera el campo desde potenciales no uniformes remuestreados a z uniforme.
    """
    coeff = prepare_coefficients_from_potentials_uniform(
        input_table,
        dz=dz,
        nz=nz,
    )

    return generate_field_map_from_coefficients(coeff, output_pkl=output_pkl)


# ============================================================
# Gráficas de diagnóstico
# ============================================================

def field_magnitude(Ex, Ey, Ez):
    return np.sqrt(Ex**2 + Ey**2 + Ez**2)


def make_diagnostic_plots(x, y, z, Ex, Ey, Ez):
    """
    Genera:

    1) plano ZX en y≈0
    2) plano YZ en x≈0
    3) plano XY en z=z1
    4) plano XY en z=z2
    """
    import matplotlib.pyplot as plt

    Emag = field_magnitude(Ex, Ey, Ez)

    ix0 = np.argmin(np.abs(x - 0.0))
    iy0 = np.argmin(np.abs(y - 0.0))

    z1 = z[int(0.25 * (len(z) - 1))]
    z2 = z[int(0.75 * (len(z) - 1))]

    iz1 = np.argmin(np.abs(z - z1))
    iz2 = np.argmin(np.abs(z - z2))

    # --------------------------------------------------------
    # Plano ZX en y=0
    # Emag[ix, iy0, iz]
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(
        z,
        x,
        Emag[:, iy0, :],
        shading="auto"
    )
    plt.xlabel("z [cm]")
    plt.ylabel("x [cm]")
    plt.title(f"Plano ZX, y={y[iy0]:.4f} cm, |E| normalizado")
    plt.colorbar(label="|E| [V/m/V]")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plane_ZX_y0_Emag.png", dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Plano YZ en x=0
    # Emag[ix0, iy, iz]
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(
        z,
        y,
        Emag[ix0, :, :],
        shading="auto"
    )
    plt.xlabel("z [cm]")
    plt.ylabel("y [cm]")
    plt.title(f"Plano YZ, x={x[ix0]:.4f} cm, |E| normalizado")
    plt.colorbar(label="|E| [V/m/V]")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plane_YZ_x0_Emag.png", dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Plano XY en z1
    # --------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(
        x,
        y,
        Emag[:, :, iz1].T,
        shading="auto"
    )
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.title(f"Plano XY, z={z[iz1]:.4f} cm, |E| normalizado")
    plt.colorbar(label="|E| [V/m/V]")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"plane_XY_z_{z[iz1]:.3f}_Emag.png", dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Plano XY en z2
    # --------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(
        x,
        y,
        Emag[:, :, iz2].T,
        shading="auto"
    )
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.title(f"Plano XY, z={z[iz2]:.4f} cm, |E| normalizado")
    plt.colorbar(label="|E| [V/m/V]")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"plane_XY_z_{z[iz2]:.3f}_Emag.png", dpi=200)
    plt.close()

    # --------------------------------------------------------
    # También graficar Ez en ZX para revisar aceleración
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(
        z,
        x,
        Ez[:, iy0, :],
        shading="auto"
    )
    plt.xlabel("z [cm]")
    plt.ylabel("x [cm]")
    plt.title(f"Plano ZX, y={y[iy0]:.4f} cm, Ez normalizado")
    plt.colorbar(label="Ez [V/m/V]")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plane_ZX_y0_Ez.png", dpi=200)
    plt.close()

    print()
    print(f"Gráficas guardadas en: {PLOT_DIR.resolve()}")
    print("  plane_ZX_y0_Emag.png")
    print("  plane_YZ_x0_Emag.png")
    print(f"  plane_XY_z_{z[iz1]:.3f}_Emag.png")
    print(f"  plane_XY_z_{z[iz2]:.3f}_Emag.png")
    print("  plane_ZX_y0_Ez.png")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if COEFFICIENT_SOURCE == "geometry":
        df_field = generate_field_map()
    elif COEFFICIENT_SOURCE == "geometry_kt":
        df_field = generate_field_map_from_geometry_kt()
    elif COEFFICIENT_SOURCE == "potentials":
        df_field = generate_field_map_from_potentials()
    elif COEFFICIENT_SOURCE == "potentials_uniform":
        df_field = generate_field_map_from_potentials_uniform()
    else:
        raise ValueError(
            "COEFFICIENT_SOURCE debe ser 'geometry', 'geometry_kt', "
            "'potentials' o 'potentials_uniform'."
        )
