#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Requisitos: pip install pandas matplotlib numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= CONFIGURA AQUÍ =========
CSV_PATH = "cross_z0p001.csv"  # <-- tu archivo
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


# --- 3) t–E (energía cinética en MeV) ---
plt.figure(figsize=(6,4))
plt.scatter(df["t_cross"], E_MeV, s=6)
plt.xlabel("t_cross [s]")
plt.ylabel(f"E_k [{E_UNIT_LABEL}]")
plt.title("Energía cinética vs tiempo de cruce")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
#plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Requisitos: pip install pandas matplotlib numpy



# === CONFIGURA AQUÍ ===
CSV1 = "cross_z0p001.csv"   # plano cercano (p. ej. z=0.001)
CSV2 = "cross_z0p300.csv"   # plano lejano (p. ej. z=0.3)
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

# Intersección por pid: solo partículas que cruzaron ambos planos
m = df1.merge(df2, on="pid", how="inner")
print(f"Total en {LABEL1}: {len(df1)}  |  Total en {LABEL2}: {len(df2)}  |  En ambos: {len(m)}")

if len(m) == 0:
    raise SystemExit("No hay pids en común entre los dos planos. Revisa tus CSVs o los filtros usados.")

# (Opcional) Ordenar por tiempo del primer cruce
m = m.sort_values("t1")

# Figuras: comparación x–vx
plt.figure(figsize=(14,4.8))

# A) x–vx en plano 1
ax1 = plt.subplot(1,3,1)
ax1.scatter(df["x"], df["vx"], s=6, alpha=0.3, label=LABEL1)
ax1.scatter(m["x1"], m["vx1"], s=6)
ax1.set_xlabel("x1 [m]")
ax1.set_ylabel("vx1 [m/s]")
ax1.set_title(f"{LABEL1}: espacio de fases x–vx")
ax1.grid(True, linestyle="--", alpha=0.4)

# B) x–vx en plano 2
ax2 = plt.subplot(1,3,2)
ax2.scatter(m["x2"], m["vx2"], s=6)
ax2.set_xlabel("x2 [m]")
ax2.set_ylabel("vx2 [m/s]")
ax2.set_title(f"{LABEL2}: espacio de fases x–vx")
ax2.grid(True, linestyle="--", alpha=0.4)

# C) Evolución: flechas (del plano 1 al 2) en el espacio x–vx
ax3 = plt.subplot(1,3,3)
x1, vx1 = m["x1"].to_numpy(),  m["vx1"].to_numpy()
x2, vx2 = m["x2"].to_numpy(),  m["vx2"].to_numpy()
dx, dvx = (x2 - x1), (vx2 - vx1)

# Para no saturar, puedes muestrear si hay demasiados puntos
MAX_ARROWS = 5000
if len(m) > MAX_ARROWS:
    idx = np.random.choice(len(m), MAX_ARROWS, replace=False)
else:
    idx = np.arange(len(m))

ax3.quiver(x1[idx], vx1[idx], dx[idx], dvx[idx], angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.7)
ax3.set_xlabel("x [m]")
ax3.set_ylabel("vx [m/s]")
ax3.set_title("Evolución (Plano 1 → Plano 2) en x–vx")
ax3.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()




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

# --- energía relativista ---
c = 299_792_458.0
#v2 = df["vx"]**2 + df["vy"]**2 + df["vz"]**2
v2 = m["vx2"]**2 + m["vy2"]**2 + m["vz2"]**2
beta2 = np.clip(v2 / c**2, 0.0, 1.0 - 1e-15)
gamma = 1.0 / np.sqrt(1.0 - beta2)
E_joules = (gamma - 1.0) * MASS_KG * c**2
E_MeV = E_joules / (1.602176634e-13)  # 1 MeV = 1.602e-13 J

# --- 3) t–E (energía cinética en MeV) ---
plt.figure(figsize=(6,4))
plt.scatter(m["t2"], E_MeV, s=6)
plt.xlabel("t_cross [s]")
plt.ylabel(f"E_k [{E_UNIT_LABEL}]")
plt.title("Energía cinética vs tiempo de cruce")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()


plt.show()

# (Opcional) Algunos resúmenes
m["dvx"] = dvx
print("Resumen Δvx [m/s]")
print(m["dvx"].describe(percentiles=[0.05,0.5,0.95]))

# Si quieres guardar tablas de comparación:
# m[["pid","t1","x1","vx1","t2","x2","vx2","dvx"]].to_csv("compare_x_vx.csv", index=False)
