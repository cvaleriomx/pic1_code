#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# Configuración
# ============================================================

WORKDIR = Path(".").resolve()
DATA_FILE = WORKDIR / "data.dat"
FORT_ENVELOPE = WORKDIR / "fort.envelope"
TEMPLATE_FORT_ENVELOPE = WORKDIR.parent.parent / "dans_1" / "fort.envelope"

OPTR_CMD = ["./optr"]

DEFAULT_PHASES_DEG = np.arange(-360.0, 0.0 + 1e-9, 5.0)
DEFAULT_VANE_VOLTAGE_MV = 0.025000000
DEFAULT_VOLTAGE_START_MV = DEFAULT_VANE_VOLTAGE_MV
DEFAULT_VOLTAGE_STOP_MV = DEFAULT_VANE_VOLTAGE_MV
DEFAULT_VOLTAGE_STEP_MV = 0.00500
DEFAULT_VOLTAGE_START_MV = 0.0250
DEFAULT_VOLTAGE_STOP_MV = 0.080
DEFAULT_POINT_TIMEOUT_SEC = 30.0

OUTPUT_DIR = WORKDIR / "scan_phase_voltage_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_NPZ = OUTPUT_DIR / "scan_phase_voltage_results.npz"


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


def load_template_columns():
    template_file = TEMPLATE_FORT_ENVELOPE if TEMPLATE_FORT_ENVELOPE.exists() else None
    if template_file is None and FORT_ENVELOPE.exists():
        template_file = FORT_ENVELOPE

    if template_file is None:
        return ["s", "E"]

    with open(template_file, "r") as f:
        columns = f.readline().strip().split()

    return columns


def set_labeled_value_in_data_dat(label, value):
    """
    Busca una línea que contenga `label` y reemplaza sólo el primer valor numérico.
    Esto conserva los límites y la bandera `varied` que ya existen en data.dat.
    """
    lines = DATA_FILE.read_text().splitlines()
    new_lines = []
    changed = False

    for line in lines:
        if label in line:
            before, sep, after = line.partition("!")
            tokens = before.split()

            if not tokens:
                raise RuntimeError(f"Encontré la línea de {label}, pero no pude parsearla.")

            tokens[0] = f"{value:.8f}"
            rebuilt = " ".join(tokens)

            if sep:
                rebuilt += f"  !{after}"

            new_lines.append(rebuilt)
            changed = True
        else:
            new_lines.append(line)

    if not changed:
        raise RuntimeError(f"No encontré la línea de {label} en data.dat.")

    DATA_FILE.write_text("\n".join(new_lines) + "\n")
    print(f"[data.dat] {label} = {value:.8f}")


def set_phase_in_data_dat(phase_deg):
    set_labeled_value_in_data_dat("Phase", phase_deg)


def set_voltage_in_data_dat(vane_voltage_mv):
    set_labeled_value_in_data_dat("Vane_voltage", vane_voltage_mv)


def make_zero_dataframe(columns):
    return pd.DataFrame({column: np.zeros(1) for column in columns})


def run_optr(timeout_sec):
    try:
        p = subprocess.run(
            OPTR_CMD,
            cwd=WORKDIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
        )
        return p.returncode, p.stdout, False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if exc.stdout is not None else ""
        return 124, stdout, True


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


def voltage_tag(voltage_mv):
    return f"v_{voltage_mv:.5f}".replace("+", "p").replace("-", "m").replace(".", "p")


def make_range(start, stop, step):
    if step <= 0:
        raise ValueError("El paso debe ser mayor que cero.")
    return np.arange(start, stop + 1e-9, step)


# ============================================================
# Corrida principal
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Barre voltaje y fase en data.dat antes de correr optr."
    )
    parser.add_argument(
        "--voltage-start-mv",
        type=float,
        default=DEFAULT_VOLTAGE_START_MV,
        help=f"Voltaje inicial en MV (default: {DEFAULT_VOLTAGE_START_MV})",
    )
    parser.add_argument(
        "--voltage-stop-mv",
        type=float,
        default=DEFAULT_VOLTAGE_STOP_MV,
        help=f"Voltaje final en MV (default: {DEFAULT_VOLTAGE_STOP_MV})",
    )
    parser.add_argument(
        "--voltage-step-mv",
        type=float,
        default=DEFAULT_VOLTAGE_STEP_MV,
        help=f"Paso del barrido de voltaje en MV (default: {DEFAULT_VOLTAGE_STEP_MV})",
    )
    parser.add_argument(
        "--phase-start",
        type=float,
        default=float(DEFAULT_PHASES_DEG[0]),
        help="Inicio del barrido de fase en grados.",
    )
    parser.add_argument(
        "--phase-stop",
        type=float,
        default=float(DEFAULT_PHASES_DEG[-1]),
        help="Fin del barrido de fase en grados.",
    )
    parser.add_argument(
        "--phase-step",
        type=float,
        default=5.0,
        help="Paso del barrido de fase en grados.",
    )
    parser.add_argument(
        "--point-timeout-sec",
        type=float,
        default=DEFAULT_POINT_TIMEOUT_SEC,
        help=f"Tiempo máximo por punto en segundos (default: {DEFAULT_POINT_TIMEOUT_SEC}).",
    )
    args = parser.parse_args()

    voltages_mv = make_range(args.voltage_start_mv, args.voltage_stop_mv, args.voltage_step_mv)
    phases_deg = make_range(args.phase_start, args.phase_stop, args.phase_step)
    template_columns = load_template_columns()

    backup_data_dat()

    final_energy_grid = np.full((len(voltages_mv), len(phases_deg)), np.nan)
    point_status_grid = np.full((len(voltages_mv), len(phases_deg)), 2, dtype=int)
    all_results = {}

    try:
        for voltage_idx, voltage_mv in enumerate(voltages_mv):
            set_voltage_in_data_dat(voltage_mv)
            voltage_results = {}

            for phase_idx, phase in enumerate(phases_deg):
                print("\n" + "=" * 70)
                print(f"Corriendo voltage {voltage_mv:.5f} MV, fase {phase:.2f} deg")
                print("=" * 70)

                set_phase_in_data_dat(phase)

                code, stdout, timed_out = run_optr(args.point_timeout_sec)
                tag = f"{voltage_tag(voltage_mv)}_{phase_tag(phase)}"

                if timed_out:
                    print(f"[timeout] voltage {voltage_mv:.5f} MV, fase {phase:.2f} deg")
                    (OUTPUT_DIR / f"{tag}_stdout.txt").write_text(
                        f"[timeout] cancelado tras {args.point_timeout_sec:.2f} s\n{stdout}"
                    )
                    df = make_zero_dataframe(template_columns)
                    point_status_grid[voltage_idx, phase_idx] = 1
                elif code != 0:
                    print(f"[error] optr falló para voltage {voltage_mv:.5f} MV, fase {phase:.2f} deg")
                    (OUTPUT_DIR / f"{tag}_stdout.txt").write_text(
                        f"[error] optr devolvió código {code}\n{stdout}"
                    )
                    df = make_zero_dataframe(template_columns)
                    point_status_grid[voltage_idx, phase_idx] = 2
                elif not FORT_ENVELOPE.exists():
                    print(f"[error] No encontré fort.envelope para voltage {voltage_mv:.5f} MV, fase {phase:.2f} deg")
                    (OUTPUT_DIR / f"{tag}_stdout.txt").write_text(
                        f"[error] No encontré fort.envelope\n{stdout}"
                    )
                    df = make_zero_dataframe(template_columns)
                    point_status_grid[voltage_idx, phase_idx] = 2
                else:
                    (OUTPUT_DIR / f"{tag}_stdout.txt").write_text(stdout)
                    df, columns, units = read_fort_envelope(FORT_ENVELOPE)
                    template_columns = list(df.columns)
                    point_status_grid[voltage_idx, phase_idx] = 0

                    shutil.copy2(
                        FORT_ENVELOPE,
                        OUTPUT_DIR / f"{tag}_fort.envelope",
                    )

                voltage_results[phase] = df
                df.to_csv(
                    OUTPUT_DIR / f"{tag}_fort_envelope.csv",
                    index=False,
                )

                if "E" in df.columns:
                    final_energy_grid[voltage_idx, phase_idx] = float(df["E"].iloc[-1])
                else:
                    final_energy_grid[voltage_idx, phase_idx] = 0.0

            all_results[voltage_mv] = voltage_results

        save_all_npz(all_results, final_energy_grid, voltages_mv, phases_deg, point_status_grid)
        make_plots(final_energy_grid, voltages_mv, phases_deg)

    finally:
        restore_data_dat()


# ============================================================
# Guardar arreglos
# ============================================================

def save_all_npz(all_results, final_energy_grid, voltages_mv, phases_deg, point_status_grid):
    save_dict = {
        "phases": phases_deg,
        "voltages_mv": voltages_mv,
        "final_energy_grid": final_energy_grid,
        "point_status_grid": point_status_grid,
    }

    for voltage_mv, voltage_results in all_results.items():
        voltage_key = f"{voltage_mv:.5f}"

        for phase, df in voltage_results.items():
            phase_key = f"{phase:.2f}"

            for col in df.columns:
                clean_col = (
                    col.replace("-", "_")
                       .replace("'", "p")
                       .replace("/", "_")
                       .replace("[", "")
                       .replace("]", "")
                )
                save_dict[f"{clean_col}_{voltage_key}_{phase_key}"] = df[col].to_numpy()

    np.savez(OUTPUT_NPZ, **save_dict)
    print(f"[save] Guardé todo en {OUTPUT_NPZ}")


# ============================================================
# Gráficas
# ============================================================

def make_plots(final_energy_grid, voltages_mv, phases_deg):
    phase_grid, voltage_grid = np.meshgrid(phases_deg, voltages_mv)

    if len(voltages_mv) > 1 and len(phases_deg) > 1:
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection="3d")
        surface = ax.plot_surface(
            voltage_grid,
            phase_grid,
            final_energy_grid,
            cmap="viridis",
            edgecolor="none",
            antialiased=True,
        )
        ax.set_xlabel("Vane voltage [MV]")
        ax.set_ylabel("Fase [deg]")
        ax.set_zlabel("Energía final [MeV]")
        ax.set_title("Superficie de energía final vs voltaje y fase")
        fig.colorbar(surface, ax=ax, shrink=0.7, pad=0.1, label="Energía final [MeV]")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "surface_energy_voltage_phase.png", dpi=200)
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(voltage_grid.ravel(), phase_grid.ravel(), c=final_energy_grid.ravel(), cmap="viridis", s=60)
        plt.xlabel("Vane voltage [MV]")
        plt.ylabel("Fase [deg]")
        plt.title("Energía final vs voltaje y fase")
        plt.colorbar(label="Energía final [MeV]")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "surface_energy_voltage_phase.png", dpi=200)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        final_energy_grid,
        origin="lower",
        aspect="auto",
        extent=[phases_deg[0], phases_deg[-1], voltages_mv[0], voltages_mv[-1]],
        cmap="viridis",
    )
    plt.xlabel("Fase [deg]")
    plt.ylabel("Vane voltage [MV]")
    plt.title("Mapa de calor de energía final")
    plt.colorbar(label="Energía final [MeV]")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_energy_voltage_phase.png", dpi=200)


if __name__ == "__main__":
    main()