import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# Leer fort.envelope
# ============================================================

filename = "fort.envelope"

with open(filename, "r") as f:
    columns = f.readline().strip().split()
    units = f.readline().strip().split()

df = pd.read_csv(
    filename,
    sep=r"\s+",
    skiprows=2,
    names=columns,
    engine="python"
)

print("Columnas disponibles:")
print(df.columns)


# ============================================================
# Funciones
# ============================================================

def covariance_from_envelopes(u_env, up_env, r_uup):
    """
    Construye matriz sigma 2x2:

        sigma = [[<u^2>, <u u'>],
                 [<u u'>, <u'^2>]]
    """

    u_env = abs(u_env)
    up_env = abs(up_env)

    sigma_uu = u_env**2
    sigma_upup = up_env**2
    sigma_uup = r_uup * u_env * up_env

    return np.array([
        [sigma_uu, sigma_uup],
        [sigma_uup, sigma_upup]
    ])


def ellipse_parameters_from_covariance(cov, nsigma=1.0):
    """
    Convierte matriz sigma 2x2 en ancho, alto y ángulo.
    """

    eigvals, eigvecs = np.linalg.eigh(cov)

    # Protección contra errores numéricos pequeños
    eigvals = np.maximum(eigvals, 0.0)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    semi_major = nsigma * np.sqrt(eigvals[0])
    semi_minor = nsigma * np.sqrt(eigvals[1])

    width = 2.0 * semi_major
    height = 2.0 * semi_minor

    vx, vy = eigvecs[:, 0]
    angle = np.degrees(np.arctan2(vy, vx))

    return width, height, angle


def make_ellipse_patch(cov, nsigma=1.0, color="C0"):
    width, height, angle = ellipse_parameters_from_covariance(
        cov,
        nsigma=nsigma
    )

    return Ellipse(
        xy=(0.0, 0.0),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=2.2,
        edgecolor=color,
    )


def rms_emittance_from_covariance(cov):
    """
    Emittancia rms geométrica:
        epsilon = sqrt(det sigma)
    """
    det = np.linalg.det(cov)
    return np.sqrt(max(det, 0.0))


# ============================================================
# Extraer columnas
# ============================================================

s = df["s"].to_numpy()

x_env = df["x-envelope"].to_numpy()
xp_env = df["x'-envelope"].to_numpy()

y_env = df["y-envelope"].to_numpy()
yp_env = df["y'-envelope"].to_numpy()

z_env = df["z-envelope"].to_numpy()
zp_env = df["z'-envelope"].to_numpy()

r12 = df["r12"].to_numpy()
r34 = df["r34"].to_numpy()
r56 = df["r56"].to_numpy()

E = df["E"].to_numpy() if "E" in df.columns else None


# ============================================================
# Reducir número de frames
# ============================================================

max_frames = 250

if len(df) > max_frames:
    frame_indices = np.linspace(0, len(df) - 1, max_frames, dtype=int)
else:
    frame_indices = np.arange(len(df))


# ============================================================
# Límites de los ejes
# ============================================================

scale = 1.30

x_lim = scale * np.nanmax(np.abs(x_env))
xp_lim = scale * np.nanmax(np.abs(xp_env))

y_lim = scale * np.nanmax(np.abs(y_env))
yp_lim = scale * np.nanmax(np.abs(yp_env))

z_lim = scale * np.nanmax(np.abs(z_env))
zp_lim = scale * np.nanmax(np.abs(zp_env))

# Evitar límites cero
x_lim = max(x_lim, 1e-8)
xp_lim = max(xp_lim, 1e-8)

y_lim = max(y_lim, 1e-8)
yp_lim = max(yp_lim, 1e-8)

z_lim = max(z_lim, 1e-8)
zp_lim = max(zp_lim, 1e-8)


# ============================================================
# Figura con tres paneles
# ============================================================

fig, (ax_x, ax_y, ax_z) = plt.subplots(1, 3, figsize=(17, 5.5))

nsigma = 1.0

title = fig.suptitle("", fontsize=14)

for ax in (ax_x, ax_y, ax_z):
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axvline(0.0, color="black", linewidth=0.7)
    ax.grid(True)

ax_x.set_xlim(-x_lim, x_lim)
ax_x.set_ylim(-xp_lim, xp_lim)
ax_x.set_xlabel("x [cm]")
ax_x.set_ylabel("x' [rad]")
ax_x.set_title("Elipse x-x'")

ax_y.set_xlim(-y_lim, y_lim)
ax_y.set_ylim(-yp_lim, yp_lim)
ax_y.set_xlabel("y [cm]")
ax_y.set_ylabel("y' [rad]")
ax_y.set_title("Elipse y-y'")

ax_z.set_xlim(-z_lim, z_lim)
ax_z.set_ylim(-zp_lim, zp_lim)
ax_z.set_xlabel("z [cm]")
ax_z.set_ylabel("z' / delta")
ax_z.set_title("Elipse longitudinal z-z'")


# ============================================================
# Objetos dinámicos
# ============================================================

ellipse_x = None
ellipse_y = None
ellipse_z = None

text_x = ax_x.text(
    0.03, 0.95, "",
    transform=ax_x.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

text_y = ax_y.text(
    0.03, 0.95, "",
    transform=ax_y.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

text_z = ax_z.text(
    0.03, 0.95, "",
    transform=ax_z.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)


# ============================================================
# Actualización de frame
# ============================================================

def update(frame_number):
    global ellipse_x, ellipse_y, ellipse_z

    idx = frame_indices[frame_number]

    # Borrar elipses anteriores
    if ellipse_x is not None:
        ellipse_x.remove()

    if ellipse_y is not None:
        ellipse_y.remove()

    if ellipse_z is not None:
        ellipse_z.remove()

    # Matrices sigma
    cov_x = covariance_from_envelopes(
        x_env[idx],
        xp_env[idx],
        r12[idx]
    )

    cov_y = covariance_from_envelopes(
        y_env[idx],
        yp_env[idx],
        r34[idx]
    )

    cov_z = covariance_from_envelopes(
        z_env[idx],
        zp_env[idx],
        r56[idx]
    )

    # Elipses
    ellipse_x = make_ellipse_patch(cov_x, nsigma=nsigma, color="C0")
    ellipse_y = make_ellipse_patch(cov_y, nsigma=nsigma, color="C1")
    ellipse_z = make_ellipse_patch(cov_z, nsigma=nsigma, color="C2")

    ax_x.add_patch(ellipse_x)
    ax_y.add_patch(ellipse_y)
    ax_z.add_patch(ellipse_z)

    # Emitancias
    emit_x = rms_emittance_from_covariance(cov_x)
    emit_y = rms_emittance_from_covariance(cov_y)
    emit_z = rms_emittance_from_covariance(cov_z)

    # Título general
    if E is not None:
        title.set_text(
            f"Evolución de elipses de emitancia   "
            f"s = {s[idx]:.2f} cm,  E = {E[idx]:.6f} MeV"
        )
    else:
        title.set_text(
            f"Evolución de elipses de emitancia   "
            f"s = {s[idx]:.2f} cm"
        )

    # Textos
    text_x.set_text(
        f"s = {s[idx]:.2f} cm\n"
        f"x_rms = {abs(x_env[idx]):.3e} cm\n"
        f"x'_rms = {abs(xp_env[idx]):.3e}\n"
        f"r12 = {r12[idx]:.3f}\n"
        f"emit_x = {emit_x:.3e}"
    )

    text_y.set_text(
        f"s = {s[idx]:.2f} cm\n"
        f"y_rms = {abs(y_env[idx]):.3e} cm\n"
        f"y'_rms = {abs(yp_env[idx]):.3e}\n"
        f"r34 = {r34[idx]:.3f}\n"
        f"emit_y = {emit_y:.3e}"
    )

    text_z.set_text(
        f"s = {s[idx]:.2f} cm\n"
        f"z_rms = {abs(z_env[idx]):.3e} cm\n"
        f"z'_rms = {abs(zp_env[idx]):.3e}\n"
        f"r56 = {r56[idx]:.3f}\n"
        f"emit_z = {emit_z:.3e}"
    )

    return (
        ellipse_x,
        ellipse_y,
        ellipse_z,
        title,
        text_x,
        text_y,
        text_z,
    )

def save_static_frame(frame_number, output_png):
    """
    Guarda una imagen estática de un frame de la animación.
    """
    update(frame_number)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"Imagen guardada como: {output_png}")


# Guardar primer y último frame
save_static_frame(0, "emittance_ellipses_initial.png")
save_static_frame(len(frame_indices) - 1, "emittance_ellipses_final.png")

# ============================================================
# Crear y guardar animación
# ============================================================

anim = FuncAnimation(
    fig,
    update,
    frames=len(frame_indices),
    interval=80,
    blit=False
)

plt.tight_layout()

output_gif = "emittance_ellipses_xyz.gif"
# Guardar primer y último frame
save_static_frame(0, "emittance_ellipses_initial.png")
save_static_frame(len(frame_indices) - 1, "emittance_ellipses_final.png")
print("max z-envelope =", np.nanmax(np.abs(df["z-envelope"])))
print("max z'-envelope =", np.nanmax(np.abs(df["z'-envelope"])))
print("min z'-envelope =", np.nanmin(df["z'-envelope"]))
print("max r56 =", np.nanmax(np.abs(df["r56"])))
print(df[["s", "z-envelope", "z'-envelope", "r56", "E"]].describe())
idx = np.nanargmax(np.abs(df["z'-envelope"]))
print("Fila de max z':")
print(df.iloc[idx][["s", "z-envelope", "z'-envelope", "r56", "E"]])

anim.save(
    output_gif,
    writer=PillowWriter(fps=5)
)

print(f"Animación guardada como: {output_gif}")

plt.show()
