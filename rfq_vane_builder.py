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

        r_x = a * ((m + 1.0) / 2.0 - (m - 1.0) / 2.0 * np.cos(2.0 * np.pi * s / L))
        r_y = a * ((m + 1.0) / 2.0 - (m - 1.0) / 2.0 * np.cos(2.0 * np.pi * s / L + np.pi))

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
):
    """Entrada única para Warp: lee el archivo y devuelve los conductores."""
    df = read_parmteq_whitespace(archivo_datos)
    profiles = build_vane_profiles(df, pts_per_cell=pts_per_cell, unit_scale=unit_scale)
    plot_vane_profiles(profiles, output_path="vane_profiles.png")
    return build_vane_conductors_from_profiles(
        profiles=profiles,
        sim_start=sim_start,
        sim_end=sim_end,
        sim_radius=sim_radius,
        vane_radius=vane_radius,
        voltage=voltage,
    )
