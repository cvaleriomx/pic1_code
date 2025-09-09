#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Requisitos: pip install pandas matplotlib numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_emit_rms(x1,xp1):
    MX=np.mean(x1)
    MPX=np.mean(xp1)
    X=x1
    PX2=xp1
    print("media = ",MX)
    rmd=np.sqrt(np.mean(x1**2))
    print("media = ",1e3*rmd)
    LS=len(xp1)
    varx=0
    varpx=0
    varxpx=0
    for i2 in range(0,len(X)):
              	varx   = varx   + (X[i2]-MX) *(X[i2]-MX)/LS
              	varpx  = varpx  + (PX2[i2]-MPX)*(PX2[i2]-MPX)/LS
              	varxpx = varxpx + (X[i2]-MX)*(PX2[i2]-MPX)/LS
    print(varx)
    e_rms = 1*np.sqrt(varx*varpx-varxpx*varxpx)
    print ("RMS Size X = %.4f mm Emittance =  %03s mm.mrad" % (np.sqrt(varx)*1e3,e_rms*1000000))
    return (e_rms)



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

degrees_to_time = lambda deg, f_rf: deg / (360.0 * f_rf)
degrees_to_z = lambda deg, f_rf, v: degrees_to_time(deg, f_rf) * v
print(degrees_to_time(360, 352e6), "s en 1 ciclo a 352 MHz")
print(degrees_to_z(360, 352e6,2936039.794467285), "m en 1 ciclo a 352 MHz y v=0.0735c")
z0lenght=degrees_to_z(360, 352e6,2936039.794467285)*4
#input("Presiona Enter para continuar...")
base="salida_nosc_continuos1/"
base="salida/"
# ========= CONFIGURA AQUÍ =========
CSV_PATH = base + "cross_z0p001.csv"  # <-- tu archivo
MASS_KG  = 1.67262192369e-27      # masa (protones por defecto)
CHARGE_C = 1.602176634e-19        # carga elemental (si la necesitas para otras cosas)
Z0_LABEL = None                   # si quieres mostrar z0 en el título, ej. "z0 = 0.1 m"
# Unidades para mostrar (solo etiquetas):
X_UNIT_LABEL  = "m"
V_UNIT_LABEL  = "m/s"
E_UNIT_LABEL  = "MeV"
# ==================================

# --- cargar CSV ---
df = pd.read_csv(CSV_PATH)

# columnas esperadas del CSV: t_cross,x,y,z,vx,vy,vz,pid[,dir]
cols_needed = ["t_cross","x","y","z","vx","vy","vz"]
for c in cols_needed:
    if c not in df.columns:
        raise ValueError(f"Falta la columna '{c}' en {CSV_PATH}")

# --- energía relativista ---
c = 299_792_458.0
v2 = df["vx"]**2 + df["vy"]**2 + df["vz"]**2
beta2 = np.clip(v2 / c**2, 0.0, 1.0 - 1e-15)
gamma = 1.0 / np.sqrt(1.0 - beta2)
E_joules = (gamma - 1.0) * MASS_KG * c**2
E_MeV = E_joules / (1.602176634e-13)  # 1 MeV = 1.602e-13 J

# --- 1) Plano x–y ---



# --- 2) Espacio de fases x–vx ---
plt.figure(figsize=(6,4))
plt.scatter(df["x"], df["vx"], s=6)
x1=df["x"]
vx1=df["vx"]
plt.xlabel(f"x [{X_UNIT_LABEL}]")
plt.ylabel(f"vx [{V_UNIT_LABEL}]")
plt.title("Espacio de fases x–vx en el cruce")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

plt.figure(figsize=(6,4))
plt.hist(df["t_cross"],bins=100)

# --- 3) t–E (energía cinética en MeV) ---
phitime11= time_to_phase_deg(df["t_cross"], f_rf=352e6, tref=df["t_cross"].min(), phi0_deg=0.0, wrap=False, center180=False)
plt.figure(figsize=(6,4))
#plt.scatter(df["t_cross"], E_MeV, s=6)
plt.scatter(phitime11, E_MeV, s=6)

plt.xlabel("t_cross [s]")
plt.ylabel(f"E_k [{E_UNIT_LABEL}]")
plt.title("Energía cinética vs tiempo de cruce")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Requisitos: pip install pandas matplotlib numpy



# === CONFIGURA AQUÍ ===
CSV1 = base + "cross_z0p001.csv"   # plano cercano (p. ej. z=0.001)
CSV2 = base + "cross_z0p300.csv"   # plano lejano (p. ej. z=0.3)
#CSV1 = base + "initial_distribution.csv"   # plano cercano (p. ej. z=0.001)
LABEL1 = "Plano 1"
LABEL2 = "Plano 2"
# =======================

def load_clean(csv_path):
    df = pd.read_csv(csv_path)
    needed = ["t_cross","x","y","z","vx","vy","vz","pid"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"{csv_path} no tiene columnas: {miss}")
    # si por alguna razón hay múltiples entradas por pid, nos quedamos con la primera aparición
    # (nuestro monitor debería escribir solo una por pid y plano)
    df = df.sort_values(["pid","t_cross"], kind="stable").drop_duplicates("pid", keep="first")
    return df

df1 = load_clean(CSV1).rename(columns={
    "t_cross":"t1","x":"x1","y":"y1","z":"z1","vx":"vx1","vy":"vy1","vz":"vz1"
})
df2 = load_clean(CSV2).rename(columns={
    "t_cross":"t2","x":"x2","y":"y2","z":"z2","vx":"vx2","vy":"vy2","vz":"vz2"
})
#df1["pid"] = df1["pid"].astype(int)
print(df1["pid"])
# Intersección por pid: solo partículas que cruzaron ambos planos
m = df1.merge(df2, on="pid", how="inner")
print(f"Total en {LABEL1}: {len(df1)}  |  Total en {LABEL2}: {len(df2)}  |  En ambos: {len(m)}")
print(len(m), "particulas en comun")
if len(m) == 0:
    raise SystemExit("No hay pids en común entre los dos planos. Revisa tus CSVs o los filtros usados.")

# (Opcional) Ordenar por tiempo del primer cruce
m = m.sort_values("t1")

#save m dataframe to csv
#m.to_csv(base + "merged_planes.csv", index=False)





# Figuras: comparación x–vx
plt.figure(figsize=(14,4.8))

# A) x–vx en plano 1
ax1 = plt.subplot(1,3,1)
ax1.scatter(df["x"], df["vx"]/df["vz"], s=6, alpha=0.3, label=LABEL1)
ax1.scatter(m["x1"], m["vx1"]/m["vz1"], s=6)
ax1.set_xlabel("x1 [m]")
ax1.set_ylabel("vx1 [m/s]")
ax1.set_title(f"{LABEL1}: espacio de fases x–vx")
ax1.grid(True, linestyle="--", alpha=0.4)

# B) x–vx en plano 2
ax2 = plt.subplot(1,3,2)
ax2.scatter(m["x2"], m["vx2"]/m["vz2"], s=6)
ax2.set_xlabel("x2 [m]")
ax2.set_ylabel("vx2 [m/s]")
ax2.set_title(f"{LABEL2}: espacio de fases x–vx")
ax2.grid(True, linestyle="--", alpha=0.4)


v2 = m["vx1"]**2 + m["vy1"]**2 + m["vz1"]**2
beta2 = np.clip(v2 / c**2, 0.0, 1.0 - 1e-15)
gamma = 1.0 / np.sqrt(1.0 - beta2)
E_joules = (gamma - 1.0) * MASS_KG * c**2
E_MeV1 = E_joules / (1.602176634e-13)  # 1 MeV = 1.602e-13 J

# --- energía relativista ---
c = 299_792_458.0
#v2 = df["vx"]**2 + df["vy"]**2 + df["vz"]**2
v2 = m["vx2"]**2 + m["vy2"]**2 + m["vz2"]**2
beta2 = np.clip(v2 / c**2, 0.0, 1.0 - 1e-15)
gamma = 1.0 / np.sqrt(1.0 - beta2)
E_joules = (gamma - 1.0) * MASS_KG * c**2
E_MeV2 = E_joules / (1.602176634e-13)  # 1 MeV = 1.602e-13 J

# C) Evolución: flechas (del plano 1 al 2) en el espacio x–vx
ax3 = plt.subplot(1,3,3)
ax3.scatter(df["t_cross"], E_MeV, s=6, alpha=0.3, label=LABEL1)
ax3.scatter(m["t1"], E_MeV1, s=6, label=LABEL1)

plt.tight_layout()

plt.figure(figsize=(14,4.8))

# A) x–vx en plano 1 (histograma 2D)
ax1 = plt.subplot(1,3,1)
#h1 = ax1.hist2d(df["x"], df["vx"], bins=200, cmap="viridis")
#h1 = ax1.hist2d(m["x1"], m["vx1"], bins=200, cmap="viridis")
# Calcular histograma 2D manualmente
H, xedges, yedges = np.histogram2d(m["x1"], m["vx1"], bins=200)

# Aplicar máscara: ocultar bins con valor = 0
Hmasked = np.ma.masked_where(H == 0, H)
# Graficar con pcolormesh
X, Y = np.meshgrid(xedges, yedges)
pcm = ax1.pcolormesh(X, Y, Hmasked.T, cmap="viridis")  # ¡Ojo el .T para alinear!
plt.colorbar(pcm, ax=ax1, label="Densidad")

#plt.colorbar(h1[3], ax=ax1, label="Densidad")
ax1.set_xlabel("x1 [m]")
ax1.set_ylabel("vx1 [m/s]")
ax1.set_title(f"{LABEL1}: espacio de fases x–vx")
ax1.grid(True, linestyle="--", alpha=0.4)

# B) x–vx en plano 2 (histograma 2D)
ax2 = plt.subplot(1,3,2)
#h2 = ax2.hist2d(m["x2"], m["vx2"], bins=200, cmap="plasma")
#plt.colorbar(h2[3], ax=ax2, label="Densidad")

H, xedges, yedges = np.histogram2d(m["x2"], m["vx2"]/m["vz2"], bins=200)
# Aplicar máscara: ocultar bins con valor = 0
Hmasked = np.ma.masked_where(H == 0, H)
# Graficar con pcolormesh
X, Y = np.meshgrid(xedges, yedges)
pcm = ax2.pcolormesh(X, Y, Hmasked.T, cmap="plasma")  # ¡Ojo el .T para alinear!
plt.colorbar(pcm, ax=ax2, label="Densidad")
ax2.set_xlabel("x2 [m]")
ax2.set_ylabel("vx2 [m/s]")
ax2.set_title(f"{LABEL2}: espacio de fases x–vx")
ax2.grid(True, linestyle="--", alpha=0.4)

# C) Evolución en t vs energía (histograma 2D)
ax3 = plt.subplot(1,3,3)
phitime= time_to_phase_deg(m["t2"], f_rf=352e6, tref=m["t2"].min(), phi0_deg=0.0, wrap=False, center180=False)

#phitime= time_to_phase_deg(recorte, f_rf=352e6, tref=recorte.min(), phi0_deg=0.0, wrap=True, center180=False)
#phitime= time_to_phase_deg(m["t1"], f_rf=352e6, tref=m["t1"].min(), phi0_deg=0.0, wrap=True, center180=False)

#phitime= m["t2"] - m["t2"].min()
H, xedges, yedges = np.histogram2d(phitime, E_MeV2, bins=500)
#range1=[[m["t1"].min(),m["t1"].max()],[0.92*E_MeV2.max(),E_MeV2.max()]]
range1=[[-20,60],[0.93*E_MeV2.max(),E_MeV2.max()]]
#range1=[[phitime.min(),1200],[0.92*E_MeV2.max(),E_MeV2.max()]]
range1=[[-100,1800],[0.93*E_MeV2.max(),E_MeV2.max()]]
H, xedges, yedges = np.histogram2d(phitime, E_MeV2, bins=800,range=range1)

# Aplicar máscara: ocultar bins con valor = 0
Hmasked = np.ma.masked_where(H == 0, H)
# Graficar con pcolormesh
X, Y = np.meshgrid(xedges, yedges)
pcm = ax3.pcolormesh(X, Y, Hmasked.T, cmap="plasma")  # ¡Ojo el .T para alinear!
plt.colorbar(pcm, ax=ax3, label="Densidad")

#h3 = ax3.hist2d(m["t1"], E_MeV1, bins=200, cmap="inferno")
#plt.colorbar(h3[3], ax=ax3, label="Densidad")
ax3.set_xlabel("Tiempo [s]")
ax3.set_ylabel("Energía [MeV]")
ax3.set_title("Evolución temporal de la energía")
ax3.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

plt.figure(figsize=(6,6))
plt.hist(m["t1"], bins=500, alpha=0.5, label=LABEL2)

plt.figure(figsize=(6,6))
plt.scatter(df["x"], df["y"], s=6)
plt.scatter(m["x1"], m["y1"], s=6, label=LABEL1)
plt.xlabel(f"x [{X_UNIT_LABEL}]")
plt.ylabel(f"y [{X_UNIT_LABEL}]")
title1 = "Impacto en el plano"
if Z0_LABEL is not None:
    title1 += f" ({Z0_LABEL})"
plt.title(title1)
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()



# --- 3) t–E (energía cinética en MeV) ---
plt.figure(figsize=(6,4))
plt.scatter(m["t2"], E_MeV2, s=6)
plt.xlabel("t_cross [s]")
plt.ylabel(f"E_k [{E_UNIT_LABEL}]")
plt.title("Energía cinética vs tiempo de cruce")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()


tmin, tmax = 4.28e-7, 4.34e-7
mask = (m["t2"] >= tmin) & (m["t2"] <= tmax) & ~np.isnan(m["t2"])

# 2) Aplica la MISMA máscara a ambos arreglos
t1_filtrado = m["t1"][mask]
t2_filtrado = m["t2"][mask]
E_MeV2_filtrado = E_MeV2[mask]
plt.figure(figsize=(6,4))
plt.scatter(t2_filtrado, E_MeV2_filtrado, s=6)


plt.figure(figsize=(6,4))
phitimecut= time_to_phase_deg(t1_filtrado, f_rf=352e6, tref=t1_filtrado.min(), phi0_deg=0.0, wrap=True, center180=False)
phitime2= time_to_phase_deg(m["t1"], f_rf=352e6, tref=m["t1"].min(), phi0_deg=0.0, wrap=False, center180=True)
phitime21= time_to_phase_deg(m["t2"], f_rf=352e6, tref=m["t2"].min(), phi0_deg=0.0, wrap=False, center180=True)

#plt.hist(phitime, bins=1000)
plt.hist(phitime2, bins=1500,alpha=0.5, label="t1phase",range=(-180,1800))
plt.hist(phitime21, bins=1500,alpha=0.5, label="t1phase cut",range=(-180,1800))
plt.legend()

transmission=len(m["t1"])/len(df["t_cross"])
print("Transmisión total:", transmission,len(m["t2"]),len(df["t_cross"]))
calc_emit_rms(m["x1"],m["vx1"]/m["vz1"])
calc_emit_rms(m["x2"],m["vx2"]/m["vz2"])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_hist2d_with_marginals(x, y, bins=100, range=None, weights=None,
                               density=False, mask_zero=False, cmap=None):
    """
    Dibuja:
      - Histograma 2D (panel central)
      - Proyección en X (panel superior)
      - Proyección en Y (panel derecho)

    Parámetros clave:
      x, y        : arrays
      bins        : int o (bins_x, bins_y)
      range       : ((xmin, xmax), (ymin, ymax)) o None
      weights     : pesos opcionales
      density     : normaliza para densidad si True
      mask_zero   : oculta celdas H==0 en el 2D si True
      cmap        : colormap de Matplotlib (opcional)
    """
    # Limpieza de NaNs
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if weights is not None:
        w = np.asarray(weights)[m]
    else:
        w = None
    x = x[m]; y = y[m]

    # Histograma 2D
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range,
                                       weights=w, density=density)

    # Proyecciones (ojo con ejes):
    # H[i, j] corresponde a bin x=i, y=j
    Hx = H.sum(axis=1)  # proyección sobre X
    Hy = H.sum(axis=0)  # proyección sobre Y

    # Mallas para pcolormesh
    X, Y = np.meshgrid(xedges, yedges, indexing="xy")

    # Layout con GridSpec
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2, width_ratios=(4, 1.2), height_ratios=(1.2, 4),
                  hspace=0.05, wspace=0.05)

    ax_top   = fig.add_subplot(gs[0, 0])  # proyección X
    ax_right = fig.add_subplot(gs[1, 1])  # proyección Y
    ax_main  = fig.add_subplot(gs[1, 0])  # histograma 2D
    ax_empty = fig.add_subplot(gs[0, 1])  # celda vacía
    ax_empty.axis("off")

    # Histograma 2D
    Hplot = np.ma.masked_where(H == 0, H) if mask_zero else H
    pcm = ax_main.pcolormesh(xedges, yedges, Hplot.T, cmap=cmap, shading="auto")
    cbar = fig.colorbar(pcm, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label("Densidad" if density else "Cuentas")

    # Proyección en X (arriba)
    ax_top.bar((xedges[:-1] + xedges[1:]) / 2.0, Hx,
               width=np.diff(xedges), align="center", edgecolor="none")
    ax_top.set_xlim(xedges[0], xedges[-1])
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("∑Y")

    # Proyección en Y (derecha)
    ax_right.barh((yedges[:-1] + yedges[1:]) / 2.0, Hy,
                  height=np.diff(yedges), align="center", edgecolor="none")
    ax_right.set_ylim(yedges[0], yedges[-1])
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlabel("∑X")

    # Sincroniza límites (útil si pasaste range)
    ax_main.set_xlim(xedges[0], xedges[-1])
    ax_main.set_ylim(yedges[0], yedges[-1])

    # Etiquetas del panel principal
    ax_main.set_xlabel("x")
    ax_main.set_ylabel("y")
    ax_main.grid(True, linestyle="--", alpha=0.3)

    return fig, (ax_main, ax_top, ax_right)

# === Ejemplo de uso ===
vp=m["vx2"]/m["vz2"]
fig, axes = plot_hist2d_with_marginals(
     m["x2"], vp,
     bins=(200, 200),
     range=((m["x2"].min(), m["x2"].max()),
            (vp.quantile(0.01),vp.quantile(0.99))),
     density=False,
     mask_zero=True
 )
# plt.show()


plt.show()
CSV_PATH = base + "initial_distribution.csv"  # <-- tu archivo
df4 = pd.read_csv(CSV_PATH)
plt.figure(figsize=(6,6))
#plt.scatter(df4["z"], df4["y"], s=6)
plt.hist(df4["z"], bins=500)
plt.show()