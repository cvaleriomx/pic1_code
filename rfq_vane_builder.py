#!/usr/bin/env python3
"""
Construcción autocontenida de conductores de RFQ desde una tabla con columnas
Cell, a, m, L y Z.

La función pública es generar_vanes(...). Devuelve únicamente los conductores
de las vanes para que el script de Warp pueda combinarlos con el resto de la
geometría.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root


# ============================================================
# Datos y lectura
# ============================================================


@dataclass
class VaneProfiles:
    z: np.ndarray
    x_plus: np.ndarray
    x_minus: np.ndarray
    y_plus: np.ndarray
    y_minus: np.ndarray


def read_parmteq_whitespace(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")

    if "Cell" in df.columns:
        df["Cell"] = df["Cell"].astype(str).str.strip()

    for col in df.columns:
        if col != "Cell":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_L_m(df: pd.DataFrame) -> pd.DataFrame:
    required = ["a", "m", "L", "Z"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    dfv = df[df["L"].notna() & df["m"].notna() & df["a"].notna() & df["Z"].notna()].copy()
    if dfv.empty:
        raise ValueError("No hay filas válidas con a, m, L y Z.")

    dfv["z_start"] = dfv["Z"] - dfv["L"]
    dfv["z_center"] = dfv["z_start"] + 0.5 * dfv["L"]
    return dfv[["Cell", "Z", "L", "m", "z_start", "z_center", "a"]].copy()


# ============================================================
# Perfil de vanes
# ============================================================


def _prepare_interpolants(df: pd.DataFrame):
    d = df.sort_values("z_center").copy()
    grouped = d.groupby("z_center", as_index=False).agg({"a": "mean", "m": "mean"})

    zc = grouped["z_center"].to_numpy()
    a_c = grouped["a"].to_numpy()
    m_c = grouped["m"].to_numpy()

    if len(zc) > 1:
        zc = zc.copy()
        for i in range(1, len(zc)):
            if zc[i] <= zc[i - 1]:
                zc[i] = zc[i - 1] + 1e-9

    ia = PchipInterpolator(zc, a_c, extrapolate=True)
    im = PchipInterpolator(zc, m_c, extrapolate=True)
    return ia, im, d


def build_vane_profiles(df: pd.DataFrame, pts_per_cell: int = 80, unit_scale: float = 0.01) -> VaneProfiles:
    """
    Construye los perfiles de las cuatro vanes.

    unit_scale convierte las unidades del archivo a metros.
    Por defecto se asume cm -> m.
    """
    if pts_per_cell <= 0:
        raise ValueError("pts_per_cell debe ser mayor que cero.")

    dfv = get_L_m(df)
    ia, im, dcell = _prepare_interpolants(dfv)

    z_all = []
    x_plus_all = []
    x_minus_all = []
    y_plus_all = []
    y_minus_all = []

    for _, row in dcell.sort_values("Z").iterrows():
        L = float(row["L"]) * unit_scale
        z_end = float(row["Z"]) * unit_scale
        z_start = z_end - L

        s = np.linspace(0.0, L, pts_per_cell, endpoint=False)
        z = z_start + s

        # Interpolación suave de a(z), m(z) usando la coordenada física.
        a = ia(z / unit_scale) * unit_scale
        m = im(z / unit_scale)
        mpr=1
        r_x = a * ((m + 1.0) / 2.0 - (m - 1.0) / 2.0 * np.cos(mpr * np.pi * s / L))
        r_y = a * ((m + 1.0) / 2.0 - (m - 1.0) / 2.0 * np.cos(mpr * np.pi * s / L + np.pi))

        z_all.append(z)
        x_plus_all.append(+r_x)
        x_minus_all.append(-r_x)
        y_plus_all.append(+r_y)
        y_minus_all.append(-r_y)

    return VaneProfiles(
        z=np.concatenate(z_all),
        x_plus=np.concatenate(x_plus_all),
        x_minus=np.concatenate(x_minus_all),
        y_plus=np.concatenate(y_plus_all),
        y_minus=np.concatenate(y_minus_all),
    )


def build_vane_profiles_rfq_helper(
    df: pd.DataFrame,
    pts_per_cell: int = 80,
    unit_scale: float = 0.01,
) -> VaneProfiles:
    """
    Construye perfiles con la convención de dos términos de py_rfq_helper.

    Cada fila representa media oscilación, con k = pi/L. La paridad de Cell
    alterna qué vane va de a a m*a y cuál va de m*a a a. La fila Cell=0 se
    usa solo como condición inicial, igual que en los archivos PARMTEQ.
    """
    if pts_per_cell <= 0:
        raise ValueError("pts_per_cell debe ser mayor que cero.")

    dfv = get_L_m(df).sort_values("Z").reset_index(drop=True)
    cell_numbers = (
        dfv["Cell"].astype(str).str.extract(r"(\d+)", expand=False)
    )
    if cell_numbers.isna().any():
        raise ValueError("No se pudo extraer el número de todas las celdas.")

    dfv["cell_number"] = cell_numbers.astype(int)

    # PARMTEQ usa Cell=0 para dar la apertura inicial, no como celda física.
    reference_row = None
    if int(dfv.iloc[0]["cell_number"]) == 0:
        reference_row = dfv.iloc[0]
        cells = dfv.iloc[1:].reset_index(drop=True)
    else:
        cells = dfv

    if cells.empty:
        raise ValueError("No hay celdas físicas para construir las vanes.")

    a_values = cells["a"].to_numpy(dtype=float) * unit_scale
    m_values = cells["m"].to_numpy(dtype=float)
    lengths = cells["L"].to_numpy(dtype=float) * unit_scale
    numbers = cells["cell_number"].to_numpy(dtype=int)

    if np.any(a_values <= 0.0) or np.any(m_values <= 0.0) or np.any(lengths <= 0.0):
        raise ValueError("Las aperturas, modulaciones y longitudes deben ser positivas.")

    if reference_row is None:
        previous_a = a_values[0]
        previous_m = m_values[0]
    else:
        previous_a = float(reference_row["a"]) * unit_scale
        previous_m = float(reference_row["m"])

    z_all = []
    x_all = []
    y_all = []
    z_cursor = 0.0

    def solve_radius(equation, initial_radius, cell_number, local_z, vane_name):
        solution = root(
            lambda radius: equation(float(radius[0])),
            np.array([initial_radius], dtype=float),
        )
        radius = float(solution.x[0])
        if not solution.success or not np.isfinite(radius) or radius <= 0.0:
            raise RuntimeError(
                f"No convergió el perfil {vane_name} en Cell={cell_number}, "
                f"z_local={local_z:.6e} m."
            )
        return radius

    for index, (a, m, length, cell_number) in enumerate(
        zip(a_values, m_values, lengths, numbers)
    ):
        if index == 0:
            prev_a = previous_a
            prev_m = previous_m
        else:
            prev_a = a_values[index - 1]
            prev_m = m_values[index - 1]

        if index + 1 < len(cells):
            next_a = a_values[index + 1]
            next_m = m_values[index + 1]
        else:
            next_a = a
            next_m = m

        a_fudge_begin = 0.5 * (1.0 + prev_a / a)
        ma_fudge_begin = 0.5 * (1.0 + prev_a * prev_m / (m * a))
        a_fudge_end = 0.5 * (1.0 + next_a / a)
        ma_fudge_end = 0.5 * (1.0 + next_a * next_m / (m * a))

        include_endpoint = index == len(cells) - 1
        sample_count = pts_per_cell + int(include_endpoint)
        local_z_values = np.linspace(
            0.0,
            length,
            sample_count,
            endpoint=include_endpoint,
        )

        k = np.pi / length
        sign = (-1.0) ** (cell_number + 1)
        x_cell = []
        y_cell = []

        for local_z in local_z_values:
            fraction = local_z / length
            a_fudge = (1.0 - fraction) * a_fudge_begin + fraction * a_fudge_end
            ma_fudge = (1.0 - fraction) * ma_fudge_begin + fraction * ma_fudge_end

            aperture = a * a_fudge
            modulation = m * ma_fudge / a_fudge
            i0_ka = np.i0(k * aperture)
            i0_kma = np.i0(k * modulation * aperture)
            denominator = modulation**2 * i0_ka + i0_kma
            a10 = (modulation**2 - 1.0) / denominator
            r0 = aperture / np.sqrt(
                1.0
                - (modulation**2 * i0_ka - i0_ka) / denominator
            )
            cos_kz = np.cos(k * local_z)

            def vane_x(radius):
                return (
                    sign * (radius / r0) ** 2
                    + a10 * np.i0(k * radius) * cos_kz
                    - sign
                )

            def vane_y(radius):
                return (
                    -sign * (radius / r0) ** 2
                    + a10 * np.i0(k * radius) * cos_kz
                    + sign
                )

            x_cell.append(
                solve_radius(vane_x, aperture, cell_number, local_z, "x")
            )
            y_cell.append(
                solve_radius(vane_y, aperture, cell_number, local_z, "y")
            )

        z_all.append(z_cursor + local_z_values)
        x_all.append(np.asarray(x_cell))
        y_all.append(np.asarray(y_cell))
        z_cursor += length

    z = np.concatenate(z_all)
    x_plus = np.concatenate(x_all)
    y_plus = np.concatenate(y_all)

    return VaneProfiles(
        z=z,
        x_plus=x_plus,
        x_minus=-x_plus,
        y_plus=y_plus,
        y_minus=-y_plus,
    )


def cosine_vane_profile_with_fudge(parameters, z_linear):
    """Build a direct cosine vane profile using smoothed a and m*a per cell."""
    vane_x = np.full_like(z_linear, np.nan, dtype=float)
    vane_y = np.full_like(z_linear, np.nan, dtype=float)

    for cell_index, cell in enumerate(parameters):
        cell_length = cell["cell length"]
        if cell_length <= 0.0:
            continue

        cell_start = cell["cumulative length"] - cell_length
        cell_end = cell["cumulative length"]
        cell_mask = (cell_start <= z_linear) & (z_linear <= cell_end)
        if not np.any(cell_mask):
            continue

        a = cell["aperture"]
        m = cell["modulation"]

        if cell_index > 0:
            prev = parameters[cell_index - 1]
            prev_a = prev["aperture"]
            prev_ma = prev_a * prev["modulation"]
            a_fudge_begin = 0.5 * (1.0 + prev_a / a)
            ma_fudge_begin = 0.5 * (1.0 + prev_ma / (m * a))
        else:
            a_fudge_begin = ma_fudge_begin = 1.0

        if cell_index + 1 < len(parameters):
            next_cell = parameters[cell_index + 1]
            next_a = next_cell["aperture"]
            next_ma = next_a * next_cell["modulation"]
            a_fudge_end = 0.5 * (1.0 + next_a / a)
            ma_fudge_end = 0.5 * (1.0 + next_ma / (m * a))
        else:
            a_fudge_end = ma_fudge_end = 1.0

        u = (z_linear[cell_mask] - cell_start) / cell_length
        smooth_u = 3.0 * u**2 - 2.0 * u**3

        a_fudge = (1.0 - smooth_u) * a_fudge_begin + smooth_u * a_fudge_end
        ma_fudge = (1.0 - smooth_u) * ma_fudge_begin + smooth_u * ma_fudge_end

        a_profile = a * a_fudge
        ma_profile = m * a * ma_fudge
        delta = ma_profile - a_profile
        c = np.cos(np.pi * u)
        sign = (-1.0) ** (cell["cell no"] + 1)

        if sign > 0.0:
            vane_x[cell_mask] = a_profile + 0.5 * delta * (1.0 - c)
            vane_y[cell_mask] = a_profile + 0.5 * delta * (1.0 + c)
        else:
            vane_x[cell_mask] = a_profile + 0.5 * delta * (1.0 + c)
            vane_y[cell_mask] = a_profile + 0.5 * delta * (1.0 - c)

    return vane_x, vane_y


def build_vane_profiles_cosine_fudge(
    df: pd.DataFrame,
    pts_per_cell: int = 80,
    unit_scale: float = 0.01,
) -> VaneProfiles:
    """
    Construye perfiles con coseno directo y fudge suave entre celdas.

    La fila Cell=0, si existe, se toma como referencia de PARMTEQ y no como
    celda física, igual que en build_vane_profiles_rfq_helper.
    """
    if pts_per_cell <= 0:
        raise ValueError("pts_per_cell debe ser mayor que cero.")

    dfv = get_L_m(df).sort_values("Z").reset_index(drop=True)
    cell_numbers = dfv["Cell"].astype(str).str.extract(r"(\d+)", expand=False)
    if cell_numbers.isna().any():
        raise ValueError("No se pudo extraer el número de todas las celdas.")

    dfv["cell_number"] = cell_numbers.astype(int)
    if int(dfv.iloc[0]["cell_number"]) == 0:
        cells = dfv.iloc[1:].reset_index(drop=True)
    else:
        cells = dfv

    if cells.empty:
        raise ValueError("No hay celdas físicas para construir las vanes.")

    a_values = cells["a"].to_numpy(dtype=float) * unit_scale
    m_values = cells["m"].to_numpy(dtype=float)
    lengths = cells["L"].to_numpy(dtype=float) * unit_scale
    numbers = cells["cell_number"].to_numpy(dtype=int)

    if np.any(a_values <= 0.0) or np.any(m_values <= 0.0) or np.any(lengths <= 0.0):
        raise ValueError("Las aperturas, modulaciones y longitudes deben ser positivas.")

    cumulative_lengths = np.cumsum(lengths)
    parameters = [
        {
            "cell no": int(cell_number),
            "cell length": float(length),
            "cumulative length": float(cumulative_length),
            "aperture": float(aperture),
            "modulation": float(modulation),
        }
        for cell_number, length, cumulative_length, aperture, modulation in zip(
            numbers,
            lengths,
            cumulative_lengths,
            a_values,
            m_values,
        )
    ]

    z_segments = []
    cell_start = 0.0
    for index, length in enumerate(lengths):
        include_endpoint = index == len(lengths) - 1
        sample_count = pts_per_cell + int(include_endpoint)
        cell_end = cell_start + length
        z_segments.append(
            np.linspace(
                cell_start,
                cell_end,
                sample_count,
                endpoint=include_endpoint,
            )
        )
        cell_start = cell_end

    z = np.concatenate(z_segments)
    x_plus, y_plus = cosine_vane_profile_with_fudge(parameters, z)

    if np.isnan(x_plus).any() or np.isnan(y_plus).any():
        raise RuntimeError("El perfil cosine_fudge dejó puntos sin calcular.")

    return VaneProfiles(
        z=z,
        x_plus=x_plus,
        x_minus=-x_plus,
        y_plus=y_plus,
        y_minus=-y_plus,
    )


# ============================================================
# Construcción Warp
# ============================================================


def union_balanced(objs):
    objs = [o for o in objs if o is not None]
    if not objs:
        return None
    if len(objs) == 1:
        return objs[0]
    mid = len(objs) // 2
    return union_balanced(objs[:mid]) + union_balanced(objs[mid:])


def build_vane_conductors_from_profiles(
    profiles: VaneProfiles,
    sim_start: float,
    sim_end: float,
    sim_radius: float,
    vane_radius: float,
    voltage: float = 1000.0,
):
    """Crea las cuatro vanes como conductores Warp.

    El tanque externo se deja fuera para que el script llamador lo combine
    con su geometría existente.
    """
    import warp as wp

    vane1_pts = np.column_stack([profiles.z, profiles.x_plus])
    vane1_npts = np.column_stack([profiles.z, profiles.x_minus])
    vane2_pts = np.column_stack([profiles.z, profiles.y_plus])
    vane2_npts = np.column_stack([profiles.z, profiles.y_minus])

    box_top_x = float(np.max(np.abs(profiles.x_plus))) + vane_radius
    box_top_y = float(np.max(np.abs(profiles.y_plus))) + vane_radius

    vane1_parts = []
    for i in range(len(vane1_pts) - 1):
        zl = vane1_pts[i + 1, 0] - vane1_pts[i, 0]
        if zl <= 0:
            continue
        zc = 0.5 * (vane1_pts[i, 0] + vane1_pts[i + 1, 0])
        yc = 0.5 * (vane1_pts[i, 1] + vane1_pts[i + 1, 1]) + vane_radius
        box_h = box_top_x - yc
        vane1_parts.append(wp.ZCylinder(vane_radius, length=zl, voltage=voltage, zcent=zc, ycent=yc))
        vane1_parts.append(wp.Box(xsize=2 * vane_radius, ysize=box_h, zsize=zl, voltage=voltage,
                                  zcent=zc, ycent=yc + 0.5 * box_h))

    vane1_cond = union_balanced(vane1_parts)

    vane1_nparts = []
    for i in range(len(vane1_npts) - 1):
        zl = vane1_npts[i + 1, 0] - vane1_npts[i, 0]
        if zl <= 0:
            continue
        zc = 0.5 * (vane1_npts[i, 0] + vane1_npts[i + 1, 0])
        yc = 0.5 * (vane1_npts[i, 1] + vane1_npts[i + 1, 1]) - vane_radius
        box_h = box_top_x - np.abs(yc)
        vane1_nparts.append(wp.ZCylinder(vane_radius, length=zl, voltage=voltage, zcent=zc, ycent=yc))
        vane1_nparts.append(wp.Box(xsize=2 * vane_radius, ysize=box_h, zsize=zl, voltage=voltage,
                                   zcent=zc, ycent=yc + 0.5 * np.sign(yc) * box_h))

    vane1_ncond = union_balanced(vane1_nparts)

    vane2_parts = []
    for i in range(len(vane2_pts) - 1):
        zl = vane2_pts[i + 1, 0] - vane2_pts[i, 0]
        if zl <= 0:
            continue
        zc = 0.5 * (vane2_pts[i, 0] + vane2_pts[i + 1, 0])
        xc = 0.5 * (vane2_pts[i, 1] + vane2_pts[i + 1, 1]) + vane_radius
        box_h = box_top_y - xc
        vane2_parts.append(wp.ZCylinder(vane_radius, length=zl, voltage=-voltage, zcent=zc, xcent=xc))
        vane2_parts.append(wp.Box(xsize=box_h, ysize=2 * vane_radius, zsize=zl, voltage=-voltage,
                                  zcent=zc, xcent=xc + 0.5 * box_h))

    vane2_cond = union_balanced(vane2_parts)

    vane2x_parts = []
    for i in range(len(vane2_npts) - 1):
        zl = vane2_npts[i + 1, 0] - vane2_npts[i, 0]
        if zl <= 0:
            continue
        zc = 0.5 * (vane2_npts[i, 0] + vane2_npts[i + 1, 0])
        xc = 0.5 * (vane2_npts[i, 1] + vane2_npts[i + 1, 1]) - vane_radius
        box_h = box_top_y - abs(xc)
        vane2x_parts.append(wp.ZCylinder(vane_radius, length=zl, voltage=-voltage, zcent=zc, xcent=xc))
        vane2x_parts.append(wp.Box(xsize=box_h, ysize=2 * vane_radius, zsize=zl, voltage=-voltage,
                                   zcent=zc, xcent=xc + 0.5 * np.sign(xc) * box_h))

    vane2x_cond = union_balanced(vane2x_parts)

    return vane1_cond + vane1_ncond + vane2_cond + vane2x_cond

# ============================================================
# crear un plot en matplotlib para verificar los perfiles
# ============================================================  
def plot_vane_profiles(profiles: VaneProfiles, output_path: str | Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(profiles.z, profiles.x_plus, label="+x")
    plt.plot(profiles.z, profiles.x_minus, label="-x")
    plt.plot(profiles.z, profiles.y_plus, "--", label="+y (fase π)")
    plt.plot(profiles.z, profiles.y_minus, "--", label="-y (fase π)")
    plt.xlabel("z (m)")
    plt.ylabel("r (m)")
    plt.title("Perfiles de las vanes")
    plt.legend()
    plt.grid()
    #opcion a mostrear el plot en pantalla:
    plt.show()
    #plt.savefig(output_path)


def generar_vanes(
    archivo_datos: str | Path,
    sim_start: float,
    sim_end: float,
    sim_radius: float,
    vane_radius: float,
    pts_per_cell: int = 80,
    unit_scale: float = 0.01,
    voltage: float = 0.0,
    profile_model: str = "legacy",
):
    """Entrada única para Warp: lee el archivo y devuelve los conductores."""
    df = read_parmteq_whitespace(archivo_datos)
    if profile_model == "legacy":
        profiles = build_vane_profiles(
            df,
            pts_per_cell=pts_per_cell,
            unit_scale=unit_scale,
        )
    elif profile_model == "rfq_helper":
        profiles = build_vane_profiles_rfq_helper(
            df,
            pts_per_cell=pts_per_cell,
            unit_scale=unit_scale,
        )
    elif profile_model == "cosine_fudge":
        profiles = build_vane_profiles_cosine_fudge(
            df,
            pts_per_cell=pts_per_cell,
            unit_scale=unit_scale,
        )
    else:
        raise ValueError(
            "profile_model debe ser 'legacy', 'rfq_helper' o 'cosine_fudge'."
        )

    plot_vane_profiles(profiles, output_path="vane_profiles.png")
    return build_vane_conductors_from_profiles(
        profiles=profiles,
        sim_start=sim_start,
        sim_end=sim_end,
        sim_radius=sim_radius,
        vane_radius=vane_radius,
        voltage=voltage,
    )
