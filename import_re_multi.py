import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.special import iv
except ImportError:
    raise ImportError("Necesitas instalar scipy: pip install scipy")


# ============================================================
# Lectura de tabla RFQ
# ============================================================

def read_rfq_cell_table(filename):
    """
    Lee tabla tipo:

    Cell V Wsyn Sig0T Sig0L A10 Phi a m B L Z A0 RFdef Oct A1

    Acepta celdas tipo:
        1R, 2R, 3R, 5, 6, ...

    Ignora encabezados y texto.
    """

    rows = []

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            clean = line.strip()

            if not clean:
                continue

            parts = clean.split()

            # La primera columna debe parecer una celda: 0, 1R, 2R, 10, etc.
            if not re.match(r"^\d+[A-Za-z]*$", parts[0]):
                continue

            cell = parts[0]

            try:
                nums = [float(x) for x in parts[1:]]
            except ValueError:
                continue

            rows.append([cell] + nums)

    if not rows:
        raise ValueError("No encontré filas numéricas en la tabla.")

    columns = [
        "Cell",
        "V_kV",
        "Wsyn_MeV",
        "Sig0T_deg",
        "Sig0L_deg",
        "A10_table",
        "Phi_deg",
        "a",
        "m",
        "B",
        "L_cm",
        "Z_cm",
        "A0_table",
        "RFdef",
        "Oct_table",
        "A1_table",
    ]

    max_len = max(len(r) for r in rows)
    df = pd.DataFrame(rows, columns=columns[:max_len])

    df["cell_number"] = (
        df["Cell"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(int)
    )

    return df


# ============================================================
# Geometría aproximada de vane
# ============================================================

def vane_radius_x(a_cm, m, kz):
    """
    Radio mínimo del par de vanes en x.

    Convención aproximada:
      rx = a en una punta
      rx = m*a en la otra
    """
    return a_cm * (1.0 + 0.5 * (m - 1.0) * (1.0 - np.cos(kz)))


def vane_radius_y(a_cm, m, kz):
    """
    Radio mínimo del par de vanes en y, desfasado media celda.
    """
    return a_cm * (1.0 + 0.5 * (m - 1.0) * (1.0 + np.cos(kz)))


def add_semicircular_vane_points(
    xs,
    ys,
    zs,
    b,
    z_local,
    r_tip,
    rho,
    vane,
    potential,
    n_arc=25,
):
    """
    Agrega puntos sobre la punta semicircular de un vane.

    Se normaliza el potencial como:

        2 Phi / V = +1  para el par de vanes x
        2 Phi / V = -1  para el par de vanes y

    vane puede ser:
        "+x", "-x", "+y", "-y"

    r_tip:
        arreglo con distancia mínima al eje para cada z

    rho:
        radio de curvatura de la punta, en cm
    """

    if vane == "+x":
        xc = r_tip + rho
        yc = np.zeros_like(r_tip)

        # Semicírculo que mira hacia el eje
        theta = np.linspace(np.pi / 2.0, 3.0 * np.pi / 2.0, n_arc)

        for th in theta:
            xs.extend(xc + rho * np.cos(th))
            ys.extend(yc + rho * np.sin(th))
            zs.extend(z_local)
            b.extend(potential * np.ones_like(z_local))

    elif vane == "-x":
        xc = -(r_tip + rho)
        yc = np.zeros_like(r_tip)

        theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_arc)

        for th in theta:
            xs.extend(xc + rho * np.cos(th))
            ys.extend(yc + rho * np.sin(th))
            zs.extend(z_local)
            b.extend(potential * np.ones_like(z_local))

    elif vane == "+y":
        xc = np.zeros_like(r_tip)
        yc = r_tip + rho

        theta = np.linspace(np.pi, 2.0 * np.pi, n_arc)

        for th in theta:
            xs.extend(xc + rho * np.cos(th))
            ys.extend(yc + rho * np.sin(th))
            zs.extend(z_local)
            b.extend(potential * np.ones_like(z_local))

    elif vane == "-y":
        xc = np.zeros_like(r_tip)
        yc = -(r_tip + rho)

        theta = np.linspace(0.0, np.pi, n_arc)

        for th in theta:
            xs.extend(xc + rho * np.cos(th))
            ys.extend(yc + rho * np.sin(th))
            zs.extend(z_local)
            b.extend(potential * np.ones_like(z_local))

    else:
        raise ValueError("vane debe ser '+x', '-x', '+y' o '-y'.")


# ============================================================
# Base multipolar
# ============================================================

def basis_functions(x, y, z, k):
    """
    Expansión usada para ajustar el potencial.

    Normalización:

        2 Phi / V =
            A01 (x^2 - y^2)
          + A10 I0(k r) cos(k z)
          + A03 r^4 cos(4 theta)
          + A12 I4(k r) cos(4 theta) cos(k z)
          + A05 r^6 cos(6 theta)

    TRANSOPTR usará solamente A01, A10, k.
    Los otros términos se guardan en el csv para diagnóstico.
    """

    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    kr = k * r
    kz = k * z

    f_A01 = x * x - y * y
    f_A10 = np.i0(kr) * np.cos(kz)

    f_A03 = r**4 * np.cos(4.0 * theta)
    f_A12 = iv(4, kr) * np.cos(4.0 * theta) * np.cos(kz)
    f_A05 = r**6 * np.cos(6.0 * theta)

    return np.vstack([
        f_A01,
        f_A10,
        f_A03,
        f_A12,
        f_A05,
    ]).T


# ============================================================
# Ajuste por celda
# ============================================================

def fit_cell_coefficients(
    a_cm,
    m,
    L_cm,
    rho_over_a=0.75,
    n_z=100,
    n_arc=25,
    ridge=1.0e-12,
):
    """
    Ajusta coeficientes multipolares de una celda usando puntas semicirculares.

    a_cm:
        apertura mínima en cm

    m:
        modulación

    L_cm:
        longitud de celda en cm

    rho_over_a:
        radio de curvatura relativo de la punta:
            rho = rho_over_a * a

    n_z:
        puntos longitudinales por celda

    n_arc:
        puntos sobre la semicircunferencia de cada vane

    ridge:
        regularización pequeña para estabilizar el ajuste
    """

    k = np.pi / L_cm

    z_local = np.linspace(0.0, L_cm, n_z)
    kz = k * z_local

    rx = vane_radius_x(a_cm, m, kz)
    ry = vane_radius_y(a_cm, m, kz)

    rho = rho_over_a * 0.01

    xs = []
    ys = []
    zs = []
    b = []

    # Par x: potencial +1
    add_semicircular_vane_points(
        xs, ys, zs, b,
        z_local=z_local,
        r_tip=rx,
        rho=rho,
        vane="+x",
        potential=+1.0,
        n_arc=n_arc,
    )

    add_semicircular_vane_points(
        xs, ys, zs, b,
        z_local=z_local,
        r_tip=rx,
        rho=rho,
        vane="-x",
        potential=+1.0,
        n_arc=n_arc,
    )

    # Par y: potencial -1
    add_semicircular_vane_points(
        xs, ys, zs, b,
        z_local=z_local,
        r_tip=ry,
        rho=rho,
        vane="+y",
        potential=-1.0,
        n_arc=n_arc,
    )

    add_semicircular_vane_points(
        xs, ys, zs, b,
        z_local=z_local,
        r_tip=ry,
        rho=rho,
        vane="-y",
        potential=-1.0,
        n_arc=n_arc,
    )

    x = np.array(xs)
    y = np.array(ys)
    z = np.array(zs)
    b = np.array(b)

    F = basis_functions(x, y, z, k)

    lhs = F.T @ F + ridge * np.eye(F.shape[1])
    rhs = F.T @ b

    coeff = np.linalg.solve(lhs, rhs)

    residual = F @ coeff - b

    rms_error = float(np.sqrt(np.mean(residual**2)))
    max_error = float(np.max(np.abs(residual)))

    names = [
        "A01_fit",
        "A10_fit",
        "A03_fit",
        "A12_fit",
        "A05_fit",
    ]

    out = dict(zip(names, coeff))

    return out, k, rms_error, max_error


# ============================================================
# Generador de fortextra75
# ============================================================

def generate_fortextra75(
    input_table="rfq_table.txt",
    output_fort="fortextra75",
    output_csv="rfq_multiterm_fit_extra75.csv",
    a_units="cm",
    use_L_column=True,
    rho_over_a=0.75,
    n_z=100,
    n_arc=25,
    ridge=1.0e-12,
    skip_cell0=True,
):
    """
    Genera archivo fortextra75 con columnas:

        s_cm   A01_fit   A10_fit   k_cm_inv

    Parámetros:
    ----------
    input_table:
        tabla RFQ original

    a_units:
        "cm" si la columna a está en cm
        "mm" si la columna a está en mm

    use_L_column:
        True  -> usa la columna L directamente
        False -> usa L = Z_i - Z_{i-1}

    rho_over_a:
        radio de punta relativo: rho = rho_over_a * a
    """

    df = read_rfq_cell_table(input_table)

    required = {"a", "m", "Z_cm"}

    if use_L_column:
        required.add("L_cm")

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = df.dropna(subset=list(required)).copy()

    if skip_cell0:
        df = df[df["cell_number"] > 0].copy()

    df = df.sort_values("cell_number").reset_index(drop=True)

    if a_units.lower() == "cm":
        df["a_cm"] = df["a"].astype(float)
    elif a_units.lower() == "mm":
        df["a_cm"] = df["a"].astype(float) / 10.0
    else:
        raise ValueError("a_units debe ser 'cm' o 'mm'.")

    z_end = df["Z_cm"].astype(float).to_numpy()

    if use_L_column:
        L_cm = df["L_cm"].astype(float).to_numpy()
    else:
        z_start = np.zeros_like(z_end)
        z_start[0] = 0.0
        z_start[1:] = z_end[:-1]
        L_cm = z_end - z_start

    if np.any(L_cm <= 0.0):
        raise ValueError("Hay longitudes de celda <= 0.")

    results = []

    for i, row in df.iterrows():
        a_cm = float(row["a_cm"])
        m = float(row["m"])
        L = float(L_cm[i])
        s_cm = float(row["Z_cm"])

        coeffs, k, rms_error, max_error = fit_cell_coefficients(
            a_cm=a_cm,
            m=m,
            L_cm=L,
            rho_over_a=rho_over_a,
            n_z=n_z,
            n_arc=n_arc,
            ridge=ridge,
        )

        out = {
            "Cell": row["Cell"],
            "cell_number": int(row["cell_number"]),
            "s_cm": s_cm,
            "L_cm_used": L,
            "a_cm": a_cm,
            "m": m,
            "rho_cm": rho_over_a * a_cm,
            "k_cm_inv": k,
            "fit_rms_error": rms_error,
            "fit_max_error": max_error,
        }

        out.update(coeffs)

        # Copiar datos de la tabla original si existen
        for col in [
            "V_kV",
            "Wsyn_MeV",
            "Sig0T_deg",
            "Sig0L_deg",
            "A10_table",
            "Phi_deg",
            "B",
            "A0_table",
            "RFdef",
            "Oct_table",
            "A1_table",
        ]:
            if col in df.columns:
                out[col] = row[col]

        results.append(out)

    out_df = pd.DataFrame(results)

    if "A10_table" in out_df.columns:
        out_df["A10_diff_fit_minus_table"] = (
            out_df["A10_fit"] - out_df["A10_table"]
        )

        out_df["A10_ratio_fit_table"] = (
            out_df["A10_fit"] / out_df["A10_table"].replace(0.0, np.nan)
        )

    # Escribir fortextra75
    # Se agrega un punto inicial s=0 con los coeficientes de la primera celda.
    with open(output_fort, "w") as f:
        first = out_df.iloc[0]

        f.write(
            f"{0.0:18.10E} "
            f"{first['A01_fit']:18.10E} "
            f"{first['A10_fit']:18.10E} "
            f"{first['k_cm_inv']:18.10E}"
            f"   ! start cell={first['Cell']}, "
            f"a_cm={first['a_cm']:.8f}, m={first['m']:.8f}, "
            f"rho_cm={first['rho_cm']:.8f}\n"
        )

        for _, row in out_df.iterrows():
            f.write(
                f"{row['s_cm']:18.10E} "
                f"{row['A01_fit']:18.10E} "
                f"{row['A10_fit']:18.10E} "
                f"{row['k_cm_inv']:18.10E}"
                f"   ! cell={row['Cell']}, "
                f"L_cm={row['L_cm_used']:.8f}, "
                f"a_cm={row['a_cm']:.8f}, "
                f"m={row['m']:.8f}, "
                f"rho_cm={row['rho_cm']:.8f}, "
                f"A03={row['A03_fit']:.6E}, "
                f"A12={row['A12_fit']:.6E}, "
                f"A05={row['A05_fit']:.6E}\n"
            )

    out_df.to_csv(output_csv, index=False)

    print()
    print(f"Archivo generado para TRANSOPTR: {output_fort}")
    print(f"Archivo de diagnóstico: {output_csv}")
    print()
    print(f"Celdas usadas       : {len(out_df)}")
    print(f"s inicial           : 0.0 cm")
    print(f"s final             : {out_df['s_cm'].iloc[-1]:.8f} cm")
    print(f"rho_over_a          : {rho_over_a}")
    print(f"n_z                 : {n_z}")
    print(f"n_arc               : {n_arc}")
    print()
    print("Primeras filas:")
    cols_to_show = [
        "Cell",
        "s_cm",
        "L_cm_used",
        "a_cm",
        "m",
        "A01_fit",
        "A10_fit",
        "k_cm_inv",
        "fit_rms_error",
    ]

    if "A10_table" in out_df.columns:
        cols_to_show.insert(7, "A10_table")

    print(out_df[cols_to_show].head(12).to_string(index=False))

    make_plots(out_df)

    return out_df


# ============================================================
# Gráficas de diagnóstico
# ============================================================

def make_plots(df):
    plot_dir = Path("fortextra75_plots")
    plot_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(df["s_cm"], df["A10_fit"], label="A10 fit semicircular", linewidth=2)

    if "A10_table" in df.columns:
        plt.plot(df["s_cm"], df["A10_table"], "--", label="A10 tabla", linewidth=2)

    plt.xlabel("s [cm]")
    plt.ylabel("A10")
    plt.title("Comparación A10")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "A10_fit_vs_table.png", dpi=200)

    plt.figure(figsize=(9, 5))
    plt.plot(df["s_cm"], df["A01_fit"], label="A01 fit", linewidth=2)

    if "A0_table" in df.columns:
        plt.plot(df["s_cm"], df["A0_table"], "--", label="A0 tabla", linewidth=2)

    plt.xlabel("s [cm]")
    plt.ylabel("coeficiente cuadrupolar")
    plt.title("A01 fit vs A0 tabla")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "A01_fit_vs_A0_table.png", dpi=200)

    plt.figure(figsize=(9, 5))
    plt.plot(df["s_cm"], df["fit_rms_error"], linewidth=2)
    plt.xlabel("s [cm]")
    plt.ylabel("RMS error")
    plt.title("Error RMS del ajuste multipolar")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / "fit_rms_error.png", dpi=200)

    if "A10_table" in df.columns:
        plt.figure(figsize=(9, 5))
        plt.plot(df["s_cm"], df["A10_diff_fit_minus_table"], linewidth=2)
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("s [cm]")
        plt.ylabel("A10_fit - A10_table")
        plt.title("Diferencia de A10")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / "A10_difference.png", dpi=200)

    plt.close("all")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    df = generate_fortextra75(
        input_table="table2_dans.txt",

        # Archivo nuevo pedido
        output_fort="fortextra75",

        # Archivo de diagnóstico
        output_csv="rfq_multiterm_fit_extra75.csv",

        # En tu tabla a parece estar en cm.
        # Si confirmas que viene en mm, cambia a "mm".
        a_units="cm",

        # True usa la columna L.
        # False usa L = Z_i - Z_{i-1}.
        use_L_column=True,

        # Radio de curvatura de la punta:
        # rho = rho_over_a * a
        rho_over_a=0.25,

        # Resolución del ajuste
        n_z=100,
        n_arc=25,

        # Regularización
        ridge=1.0e-12,

        # Ignorar celda 0 si está incompleta
        skip_cell0=True,
    )
