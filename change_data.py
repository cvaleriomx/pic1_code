#!/usr/bin/env python3
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuración
# ============================================================

WORKDIR = Path(".").resolve()
DATA_FILE = WORKDIR / "data.dat"
FORT_ENVELOPE = WORKDIR / "fort.envelope"

OPTR_CMD = ["./optr"]

PHASES_DEG = np.arange(-140.0, -10.0 + 1e-9, 10.0)
PHASES_DEG = np.arange(-180.0, -100.0 + 1e-9, 5.0)
PHASES_DEG = np.arange(-360.0, 0.0 + 1e-9, 5.0)
OUTPUT_DIR = WORKDIR / "scan_phase_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_NPZ = OUTPUT_DIR / "scan_phase_results.npz"


# ============================================================
# Utilidades
# ============================================================

def backup_data_dat():
    backup = OUTPUT_DIR / "data.dat.original"
    if not backup.exists():
        shutil.copy2(DATA_FILE, backup)
        print(f"[backup] Guardé respaldo en {backup}")


def restore_data_dat():
    backup = OUTPUT_DIR / "data.dat.original"
    if backup.exists():
        shutil.copy2(backup, DATA_FILE)
        print("[restore] data.dat restaurado")


def set_phase_in_data_dat(phase_deg):
    """
    Busca la línea que contiene 'Phase' y 'Deg' y reemplaza:
    valor lower upper varied
    por:
    phase -360 360 0

    El 0 final fija la fase.
    """
    lines = DATA_FILE.read_text().splitlines()
    new_lines = []
    changed = False

    for line in lines:
        if "Phase" in line and "Deg" in line:
            comment = ""
            if "!" in line:
                comment = "!" + line.split("!", 1)[1]

            new_line = f" {phase_deg: .8f}  -360.0  360.0   0   {comment}"
            new_lines.append(new_line)
            changed = True
        else:
            new_lines.append(line)

    if not changed:
        raise RuntimeError("No encontré la línea de Phase en data.dat.")

    DATA_FILE.write_text("\n".join(new_lines) + "\n")
    print(f"[data.dat] Phase = {phase_deg:.2f} deg")


def run_optr():
    p = subprocess.run(
        OPTR_CMD,
        cwd=WORKDIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return p.returncode, p.stdout


def read_fort_envelope(filename):
    """
    Lee fort.envelope usando el mismo método que tu código.
    """
    filename = Path(filename)

    with open(filename, "r") as f:
        columns = f.readline().strip().split()
        units = f.readline().strip().split()

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=2,
        names=columns,
        engine="python",
    )

    return df, columns, units


def phase_tag(phase):
    return f"phase_{phase:+07.2f}".replace("+", "p").replace("-", "m").replace(".", "p")


# ============================================================
# Corrida principal
# ============================================================

def main():
    backup_data_dat()

    results = {}
    final_energy = []

    try:
        for phase in PHASES_DEG:
            print("\n" + "=" * 70)
            print(f"Corriendo fase {phase:.2f} deg")
            print("=" * 70)

            set_phase_in_data_dat(phase)

            code, stdout = run_optr()
            tag = phase_tag(phase)

            (OUTPUT_DIR / f"{tag}_stdout.txt").write_text(stdout)

            if code != 0:
                print(stdout)
                raise RuntimeError(f"optr falló para fase {phase}")

            if not FORT_ENVELOPE.exists():
                raise RuntimeError("No encontré fort.envelope. Revisa IPRINT=-1.")

            df, columns, units = read_fort_envelope(FORT_ENVELOPE)

            # Guardar fort.envelope de esta fase
            df.to_csv(
                OUTPUT_DIR / f"{tag}_fort_envelope.csv",
                index=False,
            )

            # También guardarlo como texto original
            shutil.copy2(
                FORT_ENVELOPE,
                OUTPUT_DIR / f"{tag}_fort.envelope",
            )

            results[phase] = df

            if "E" in df.columns:
                final_energy.append([phase, df["E"].iloc[-1]])
            else:
                raise RuntimeError("No encontré columna E en fort.envelope.")

        save_all_npz(results, np.array(final_energy))
        make_plots(results, np.array(final_energy))

    finally:
        restore_data_dat()


# ============================================================
# Guardar arreglos
# ============================================================

def save_all_npz(results, final_energy):
    save_dict = {
        "phases": PHASES_DEG,
        "final_energy": final_energy,
    }

    for phase, df in results.items():
        key = f"{phase:.2f}"

        for col in df.columns:
            clean_col = (
                col.replace("-", "_")
                   .replace("'", "p")
                   .replace("/", "_")
                   .replace("[", "")
                   .replace("]", "")
            )
            save_dict[f"{clean_col}_{key}"] = df[col].to_numpy()

    np.savez(OUTPUT_NPZ, **save_dict)
    print(f"[save] Guardé todo en {OUTPUT_NPZ}")


# ============================================================
# Gráficas
# ============================================================

def make_plots(results, final_energy):
    # --------------------------------------------------------
    # Energía vs s para cada fase
    # --------------------------------------------------------
    plt.figure(figsize=(11, 6))

    for phase, df in results.items():
        plt.plot(df["s"], df["E"], label=f"{phase:.0f} deg")

    plt.xlabel("s [cm]")
    plt.ylabel("E [MeV]")
    plt.title("Energía a lo largo de la RFQ para cada fase")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "energia_vs_s_todas_las_fases.png", dpi=200)

    # --------------------------------------------------------
    # Energía final vs fase
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(final_energy[:, 0], final_energy[:, 1], marker="o")
    plt.xlabel("Fase [deg]")
    plt.ylabel("Energía final [MeV]")
    plt.title("Energía final vs fase")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "energia_final_vs_fase.png", dpi=200)

    # --------------------------------------------------------
    # x-envelope para cada fase
    # --------------------------------------------------------
    plt.figure(figsize=(11, 6))

    for phase, df in results.items():
        plt.plot(df["s"], df["x-envelope"], label=f"{phase:.0f} deg")

    plt.xlabel("s [cm]")
    plt.ylabel("x-envelope [cm]")
    plt.title("x-envelope para cada fase")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "x_envelope_todas_las_fases.png", dpi=200)

    # --------------------------------------------------------
    # y-envelope para cada fase
    # --------------------------------------------------------
    plt.figure(figsize=(11, 6))

    for phase, df in results.items():
        # Uso abs porque tú lo graficas como -df["y-envelope"].
        # Si quieres mantener el signo original, cambia abs(...) por df["y-envelope"].
        plt.plot(df["s"], np.abs(df["y-envelope"]), label=f"{phase:.0f} deg")

    plt.xlabel("s [cm]")
    plt.ylabel("|y-envelope| [cm]")
    plt.title("y-envelope para cada fase")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "y_envelope_todas_las_fases.png", dpi=200)

    # --------------------------------------------------------
    # z-envelope para cada fase
    # --------------------------------------------------------
    plt.figure(figsize=(11, 6))

    for phase, df in results.items():
        plt.plot(df["s"], df["z-envelope"], label=f"{phase:.0f} deg")

    plt.xlabel("s [cm]")
    plt.ylabel("z-envelope [cm]")
    plt.title("z-envelope para cada fase")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "z_envelope_todas_las_fases.png", dpi=200)

    # --------------------------------------------------------
    # z'-envelope para cada fase
    # --------------------------------------------------------
    plt.figure(figsize=(11, 6))

    for phase, df in results.items():
        plt.plot(df["s"], df["z'-envelope"], label=f"{phase:.0f} deg")

    plt.xlabel("s [cm]")
    plt.ylabel("z'-envelope")
    plt.title("z'-envelope para cada fase")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "zp_envelope_todas_las_fases.png", dpi=200)

    plt.show()

    print(f"[plot] Gráficas guardadas en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
