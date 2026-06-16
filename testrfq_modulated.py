from warp import *
from py_rfq_helper.py_rfq_helper import *
from py_rfq_helper.py_rfq_designer import *
from py_rfq_helper.py_rfq_utils import *
from py_rfq_helper.field_utils import FieldGenerator

import bisect
import time
import pprint
from pathlib import Path
from dans_pymodules import IonSpecies, ParticleDistribution, FileDialog, MyColors
import numpy as np
import scipy.constants as const
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtCore import QThread
from random import sample
from mpi4py import MPI
from my_pzplots import *
import matplotlib.pyplot as plt
from rfq_plots_library import *
def info(name, x):
    print(f"\n{name}:")
    print("  type:", type(x))
    if hasattr(x, "ndim"):
        print("  ndim:", x.ndim)
    if hasattr(x, "shape"):
        print("  shape:", x.shape)
    else:
        print("  shape: (no tiene .shape)")
    # extra útil para arrays:
    if hasattr(x, "dtype"):
        print("  dtype:", x.dtype)

__author__ = "Jared Hwang"
__doc__ = """Example PyRFQ Simulation"""




colors = MyColors()

def cosine_vane_profile_with_fudge(parameters, z_linear):
    """Build a direct cosine vane profile using smoothed a and m*a per cell."""
    vane_x = np.full_like(z_linear, np.nan, dtype=float)
    vane_y = np.full_like(z_linear, np.nan, dtype=float)

    for cell_number, cell in enumerate(parameters):
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

        if 0 < cell_number:
            prev_a = parameters["aperture"][cell_number - 1]
            prev_ma = prev_a * parameters["modulation"][cell_number - 1]
            a_fudge_begin = 0.5 * (1.0 + prev_a / a)
            ma_fudge_begin = 0.5 * (1.0 + prev_ma / (m * a))
        else:
            a_fudge_begin = ma_fudge_begin = 1.0

        if (cell_number + 1) < len(parameters):
            next_a = parameters["aperture"][cell_number + 1]
            next_ma = next_a * parameters["modulation"][cell_number + 1]
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

def load_from_ibsimu(filename):
    # Some constants
    clight = const.value("speed of light in vacuum")  # (m/s)
    amu_kg = const.value("atomic mass constant")  # (kg)
    amu_mev = const.value("atomic mass constant energy equivalent in MeV")  # MeV
    echarge = const.value("elementary charge")

    # IBSimu particle file: I, M (kg), t, x (m), vx (m/s), y (m), vy (m/s), z (m), vz (m/s)
    with open(filename) as infile:
        lines = infile.readlines()

    npart = len(lines)

    current = np.empty(npart)
    mass = np.empty(npart)
    x = np.empty(npart)
    y = np.empty(npart)
    z = np.empty(npart)
    vx = np.empty(npart)
    vy = np.empty(npart)
    vz = np.empty(npart)

    for i, line in enumerate(lines):
        current[i], mass[i], _, x[i], vx[i], y[i], vy[i], z[i], vz[i] = [float(item) for item in line.strip().split()]

    masses = np.sort(np.unique(mass))  # mass in MeV, sorted in ascending order (protons before h2+)

    particle_distributions = []

    for i, m in enumerate(masses):

        m_mev = m / amu_kg * amu_mev

        species_indices = np.where((mass == m) & (vz > 5.0e5))

        ion = IonSpecies("Species {}".format(i + 1),
                         mass_mev=m_mev,
                         a=m_mev / amu_mev,
                         z=np.round(m_mev / amu_mev, 0),
                         q=1.0,
                         current=np.sum(current[species_indices]),
                         energy_mev=1)  # Note: Set energy to 1 for now, will be recalculated when calling emittance

        particle_distributions.append(
            ParticleDistribution(ion=ion,
                                 x=x[species_indices],
                                 y=y[species_indices],
                                 z=z[species_indices],
                                 vx=vx[species_indices],
                                 vy=vy[species_indices],
                                 vz=vz[species_indices]
                                 ))


        # plt.scatter(x[species_indices], y[species_indices], s=0.5)
        # plt.show()
        # plt.scatter(x[species_indices], vx[species_indices]/vz[species_indices], s=0.5)
        # plt.show()

        particle_distributions[-1].calculate_emittances()


    return current, mass, x, vx, y, vy, z, vz





def main():
    # Initialize field generator and load field parameters
    out_field  = "input/generated_from_parmteq.dat"
    parmteqout = "input/PARMTEQOUT.TXT"
    RF_FREQ    = 32.8e6

    fg = FieldGenerator(resolution=0.0005, xy_limits=(-0.01, 0.01, -0.01, 0.01))
    fg._voltage = 22e3
    fg._frequency = RF_FREQ
    fg._a_init = 0.038802
    fg.load_parameters_from_file(parmteqout)
    fg.set_calculate_vane_profile(True)
    fg.generate()
    fg.save_field_to_file(out_field)

    # Initialization of basic RFQ parameters
    VANE_RAD   = 1.0 * cm  # radius of vane cylinder
    VANE_DIST  = 2.5 * cm  # distance of vane center to central axis
    NX, NY, NZ = 26, 26, 512
    PRWALL     = 0.04
    D_T        = 1e-9
    Z_START    = 0.01  #t he start of the rfq
    SIM_START  = -0.014
    lambda_rf  = const.c / RF_FREQ
    setup() # Warp setup function
    top.ssnpid = nextpid()
    top.npid = nextpid()

    ## Warp parameter specifications for simulation
    w3d.solvergeom = w3d.XYZgeom

    w3d.xmmax =  PRWALL
    w3d.xmmin = -PRWALL
    w3d.nx    =  NX

    w3d.ymmax =  PRWALL
    w3d.ymmin = -PRWALL
    w3d.ny    =  NY

    w3d.zmmax =  1.456 + 0.3
    w3d.zmmin =  SIM_START
    w3d.nz    =  NZ

    w3d.bound0   = neumann
    w3d.boundnz  = neumann
    w3d.boundxy  = neumann
    # ---   for particles
    top.pbound0  = absorb
    top.pboundnz = absorb
    top.prwall   = PRWALL

    top.dt = D_T

    # refinedsolver = MRBlock3D()  # Refined mesh solver
    # registersolver(refinedsolver)
    solver = MultiGrid3D()  # Non-refined mesh solver
    registersolver(solver)

    top.npinject = 50
    top.inject   = 1
    w3d.l_inj_rz = False
    top.zinject  = SIM_START
    w3d.zmmin    = SIM_START
    top.injctspc = 1000000

    ## RFQ specification and declaration
    #rfq = PyRFQ(filename=FIELD_FILENAME, from_cells=False, twoterm=False, boundarymethod=False)
    rfq = PyRFQ(filename=out_field,sim_sta=SIM_START, sim_end=1.456 + 0.3, sim_radius=PRWALL, voltage=25e3)
    #def __(self, filename, sim_sta, sim_end, sim_radius, voltage=None, debug=False):

    rfq.vane_radius    = VANE_RAD
    rfq.vane_distance  = VANE_DIST
    rfq.zstart         = Z_START
    rfq.rf_freq        = RF_FREQ
    rfq.sim_start      = SIM_START
    rfq.sim_end_buffer = 0.5
    rfq.resolution     = 0.002
    rfq.endplates      = False
    rfq.field_scaling_factor = 2
    rfq.vane_from_profile = True
    rfq.tank_from_data = False
    rfq.xy_limits = [-0.03, 0.03, -0.03, 0.03]
    rfq.z_limits  = [0, 1.5]
    rfq._voltage  = 22e3
    rfq.tt_a_init = 0.038802

    profile_mask = (
        np.isfinite(fg._z_linear)
        & np.isfinite(fg._vane_profile_x)
        & np.isfinite(fg._vane_profile_y)
        & (fg._vane_profile_x > 0.0)
        & (fg._vane_profile_y > 0.0)
    )
    if np.count_nonzero(profile_mask) < 2:
        raise RuntimeError("No se pudo generar un perfil de vanes válido desde PARMTEQOUT.TXT.")

    z_profile = fg._z_linear[profile_mask] + Z_START
    vane1_pts = np.column_stack([z_profile, fg._vane_profile_y[profile_mask]])
    vane2_pts = np.column_stack([z_profile, fg._vane_profile_x[profile_mask]])

    print("Using vane profile from", parmteqout)
    print("Vane profile points:", len(z_profile))
    print("Vane z range: {:.6f} m to {:.6f} m".format(z_profile[0], z_profile[-1]))

    cos_vane_x, cos_vane_y = cosine_vane_profile_with_fudge(
        fg._parameters,
        fg._z_linear,
    )

    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    cell_edges = fg._parameters["cumulative length"] + Z_START
    cell_edges = cell_edges[(z_profile[0] <= cell_edges) & (cell_edges <= z_profile[-1])]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(z_profile, 1.0e3 * fg._vane_profile_x[profile_mask],
                 label="x equipotencial", linewidth=1.8)
    axes[0].plot(z_profile, 1.0e3 * fg._vane_profile_y[profile_mask],
                 label="y equipotencial", linewidth=1.8)
    axes[0].plot(z_profile, 1.0e3 * cos_vane_x[profile_mask],
                 "--", label="x coseno + fudge", linewidth=1.3)
    axes[0].plot(z_profile, 1.0e3 * cos_vane_y[profile_mask],
                 "--", label="y coseno + fudge", linewidth=1.3)
    for edge in cell_edges:
        axes[0].axvline(edge, color="0.85", linewidth=0.5)
    axes[0].set_ylabel("distancia al eje [mm]")
    axes[0].set_title("Modulacion de vanes desde PARMTEQOUT")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(z_profile, 1.0e3 * (
        fg._vane_profile_x[profile_mask] - fg._vane_profile_y[profile_mask]
    ), label="equipotencial", linewidth=1.8)
    axes[1].plot(z_profile, 1.0e3 * (
        cos_vane_x[profile_mask] - cos_vane_y[profile_mask]
    ), "--", label="coseno + fudge", linewidth=1.3)
    for edge in cell_edges:
        axes[1].axvline(edge, color="0.85", linewidth=0.5)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("x - y [mm]")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(z_profile, 1.0e3 * (
        cos_vane_x[profile_mask] - fg._vane_profile_x[profile_mask]
    ), label="x coseno - x equip.", linewidth=1.5)
    axes[2].plot(z_profile, 1.0e3 * (
        cos_vane_y[profile_mask] - fg._vane_profile_y[profile_mask]
    ), label="y coseno - y equip.", linewidth=1.5)
    for edge in cell_edges:
        axes[2].axvline(edge, color="0.85", linewidth=0.5)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("z [m]")
    axes[2].set_ylabel("diferencia [mm]")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    plt.show()
    fig.savefig(plot_dir / "vane_modulation_from_parmteq.png", dpi=200)
    plt.close(fig)

    rfq.vane_top_abs = 0.03
    rfq.vane_profile = (vane1_pts, vane2_pts)
    #self.vane_from_profile = True

    # rfq.add_endplates  = True
    # rfq.cyl_id         = 0.1
    # rfq.grid_res_bempp = 0.005
    # rfq.pot_shift      = 3.0 * 22000.0
    # rfq.ignore_rms  = False
    #rfq.simple_rods = True
    rfq.simple_rods = False
    #rfq.create_vanesonductors

    rfq.setup()
    rfq.install()
    crt1 = rfq._conductors
    print(crt1.__class__.__name__, repr(crt1.name))

    ##################################### WARP BEAM
    # beam = Species(type=Dihydrogen, charge_state=pd[1].ion.z(), name=pd[1].ion.name())
    # beam = Species(type=Dihydrogen, charge_state=+1, name="H2+", color=red)

    # beam.ekin  = 15.*kV      # ion kinetic energy [eV] [eV]
    # beam.ibeam = 10 * mA  # compensated beam current [A]
    # beam.emitx = 1.8e-6  # beam x-emittance, rms edge [m-rad]
    # beam.emity = 1.8e-6  # beam y-emittance, rms edge [m-rad]
    # beam.vthz  = 0.0  # axial velocity spread [m/s ec]

    # # Beam centroid and envelope initial conditions

    # twiss_emitx = 1.8e-6 /6
    # twiss_emity = 1.8e-6 /6
    # twiss_alphax = 1.9896856 #dimensionless
    # twiss_alphay = 1.9896856
    # twiss_alphaz = 0
    # twiss_betax = 13.241259 *cm / mm # cm/mrad
    # twiss_betay = 13.241259 *cm / mm
    # twiss_betaz = 45
    # twiss_gammax = (1 + twiss_alphax**2) / twiss_betax
    # twiss_gammay = (1 + twiss_alphay**2) / twiss_betay

    # beamxangle = -sqrt(twiss_emitx * twiss_gammax)
    # beamx = sqrt(twiss_emitx * twiss_betax)
    # beamyangle = -sqrt(twiss_emity * twiss_gammay)
    # beamy = sqrt(twiss_emity * twiss_betay)

    # beam.x0  = 0.0  # initial x-centroid xc = <x> [m]
    # beam.y0  = 0.0  # initial y-centroid yc = <y> [m]
    # beam.xp0 = 0.0  # initial x-centroid angle xc' = <x'> = d<x>/ds [rad]
    # beam.yp0 = 0.0  # initial y-centroid angle yc' = <y'> = d<y>/ds [rad]
    # beam.a0  = beamx
    # beam.b0  = beamy
    # beam.ap0 = beamxangle
    # beam.bp0 = beamyangle


    # beam.a0  = 5 * mm  # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
    # beam.b0  = 5 * mm  # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
    # beam.ap0 = -0.06 # initial x-envelope angle ap = a' = d a/ds [rad]
    # beam.bp0 = -0.06  # initial y-envelope angle bp = b' = d b/ds [rad]
    ###########################################################


    ##################################### PARTICLE DISTRIBUTION
    current, mass, x, vx, y, vy, z, vz = load_from_ibsimu('input/particle_out_461mm_n5kv_10ma_20KV.txt')
    all_particles = np.array(list(zip(current, mass, x, vx, y, vy, z, vz)))

    h2_list = all_particles[np.where(mass > mass.min())]
    proton_list = all_particles[np.where(mass == mass.min())]

    h2_num = len(h2_list)
    proton_num = len(proton_list)

    h2_current = sum([i for i, _, _, _, _, _, _, _ in h2_list])
    proton_current = sum([i for i, _, _, _, _, _, _, _ in proton_list])

    h2_beam = Species(type=Dihydrogen, charge_state=+1, name="H2_1+", color=blue)
    proton_beam = Species(type=Proton, charge_state=+1, name="P", color=red)

    top.ainject = 0.05
    top.binject = 0.05
    h2_beam.ibeam = h2_current
    proton_beam.ibeam = proton_current
    h2_beam.ekin = 15.*kV
    proton_beam.ekin = 15.*kV

    # beam.ekin  = 15.*kV      # ion kinetic energy [eV] [eV]
    # beam.ibeam = 10 * mA  # compensated beam current [A]
    # beam.emitx = 1e-6  # beam x-emittance, rms edge [m-rad]
    # beam.emity = 1e-6  # beam y-emittance, rms edge [m-rad]
    # beam.vthz  = 0.0  # axial velocity spread [m/s ec]

    # injection required flags
    w3d.l_inj_user_particles_v = true
    top.linj_enormcl = false
    top.linj_efromgrid = true

    h2_beam_id = 0
    proton_beam_id = 1

    # Adding in amount of particles proportional to total number in distribution
    total_parts_per_step = 500
    h2_per_step = int(total_parts_per_step * (h2_num / (h2_num + proton_num)))
    prot_per_step = int(total_parts_per_step * (proton_num / (h2_num + proton_num)))

    def injectionsource():
        if (w3d.inj_js == h2_beam_id):
            nump = h2_per_step
            w3d.npgrp = nump
            gchange('Setpwork3d')

            idx = np.random.choice(np.arange(len(h2_list)), h2_per_step, replace=False)
            h2_inject = h2_list[idx]
            _, _, h2_x, h2_vx, h2_y, h2_vy, h2_z, h2_vz = list(zip(*h2_inject))
            w3d.xt[:] = h2_x
            w3d.yt[:] = h2_y
            # w3d.zt[:] = np.full((len(h2_x)), 0)
            w3d.uxt[:] = h2_vx
            w3d.uyt[:] = h2_vy
            w3d.uzt[:] = h2_vz

        elif (w3d.inj_js == proton_beam_id):
            nump = prot_per_step
            w3d.npgrp = nump
            gchange('Setpwork3d')

            idx = np.random.choice(np.arange(len(proton_list)), prot_per_step, replace=False)
            proton_inject = proton_list[idx]
            _, _, p_x, p_vx, p_y, p_vy, p_z, p_vz = list(zip(*proton_inject))
            w3d.xt[:] = p_x
            w3d.yt[:] = p_y
            # w3d.zt[:] = np.full((len(p_x)), 0)
            w3d.uxt[:] = p_vx
            w3d.uyt[:] = p_vy
            w3d.uzt[:] = p_vz

    installuserparticlesinjection(injectionsource)


    # Beam centroid and envelope initial conditions
    h2_beam.x0  = 0.0  # initial x-centroid xc = <x> [m]
    h2_beam.y0  = 0.0  # initial y-centroid yc = <y> [m]
    h2_beam.xp0 = 0.0  # initial x-centroid angle xc' = <x'> = d<x>/ds [rad]
    h2_beam.yp0 = 0.0  # initial y-centroid angle yc' = <y'> = d<y>/ds [rad]
    h2_beam.a0  = 5 * mm  # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
    h2_beam.b0  = 5 * mm  # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
    h2_beam.ap0 = -0.03 # initial x-envelope angle ap = a' = d a/ds [rad]
    h2_beam.bp0 = -0.03 # initial y-envelope angle bp = b' = d b/ds [rad]
    h2_beam.ekin = 15 * kV

    proton_beam.x0  = 0.0  # initial x-centroid xc = <x> [m]
    proton_beam.y0  = 0.0  # initial y-centroid yc = <y> [m]
    proton_beam.xp0 = 0.0  # initial x-centroid angle xc' = <x'> = d<x>/ds [rad]
    proton_beam.yp0 = 0.0  # initial y-centroid angle yc' = <y'> = d<y>/ds [rad]
    proton_beam.a0  = 5 * mm  # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
    proton_beam.b0  = 5 * mm  # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
    proton_beam.ap0 = -0.03 # initial x-envelope angle ap = a' = d a/ds [rad]
    proton_beam.bp0 = -0.03 # initial y-envelope angle bp = b' = d b/ds [rad]
    proton_beam.ekin = 15 * kV
    ############################################################################


    top.lrelativ = False
    top.zbeam = 0.0
    top.pgroup.nps = 0

    utils = PyRfqUtils(rfq, [h2_beam, proton_beam])


    ##################################### MESH REFINEMENT
    # note: after testing, using mesh refinement appeared to be slower than not using it, ergo commented out
    # boundaries = utils.find_vane_mesh_boundaries(NX, SIM_START, w3d.zmmax, -PRWALL, PRWALL, VANE_DIST, VANE_RAD)


    # northvane_mesh = refinedsolver.addchild(mins=boundaries["northmins"],
    #                                         maxs=boundaries["northmaxs"],
    #                                         refinement=[4,4,4])

    # southvane_mesh = refinedsolver.addchild(mins=boundaries["southmins"],
    #                                         maxs=boundaries["southmaxs"],
    #                                         refinement=[4,4,4])

    # westvane_mesh  = refinedsolver.addchild(mins=boundaries["westmins"],
    #                                         maxs=boundaries["westmaxs"],
    #                                         refinement=[4,4,4])

    # eastvane_mesh  = refinedsolver.addchild(mins=boundaries["eastmins"],
    #                                         maxs=boundaries["eastmaxs"],
    #                                         refinement=[4,4,4])

    # # Mesh refinement for the center of the beam
    # childmesh = refinedsolver.addchild(mins=[-VANE_DIST+VANE_RAD, -VANE_DIST+VANE_RAD, SIM_START],
    #                                    maxs=[ VANE_DIST-VANE_RAD,  VANE_DIST-VANE_RAD, w3d.zmmax],
    #                                    refinement=[4,4,4])
    #########################################

    derivqty()

    package("w3d")
    generate()

    # WARP built in plotting
    # @callfromafterstep
    # def makeplots():
    #     if top.it > 19900:
    #         if top.it%10 == 0:
    #             # utils.plot_rms()
    #             utils.beamplots()
    #             # print(h2_beam.getux())
    #             # window()
    #             # limits(-0.1, 1, -0.01, 0.01)
    #             # pzxedges(color='blue')
    #             # pzyedges(color='red')
    #             # fma()
    #             # refresh()

    ################################# PyQTGraph RMS plotting
    # PyQtgraph setup
    app = pg.mkQApp()

    # Setup the rms plot
    utils.rms_plot_setup(title="X and Y RMS (twice rms) vs Z", labels={'left':('X, Y', 'm'), 'bottom':('Z', 'm')},
                         xrange=[-0.1, 1.6], yrange=[-0.015, 0.015])

    # ## setup the particle plots. Not recommended, slows down simulation immensely
    # # utils.particle_plot_setup(title="X and Y Particles vs Z", labels={'left':('X, Y', 'm'), 'bottom':('Z', 'm')})

    @callfromafterstep
    def plotpyqt():
        #if top.it%2 == 1:
        #    utils.plot_rms()
        if False:
            #top.it%200 == 1:
            ppp = getphi(iy=int(w3d.ny/2),solver=getregisteredsolver())
            curr1 = top.curr
            zplmesh = top.zplmesh
            curr1 = curr1.ravel()  # O también puedes usar y.reshape(-1)
        
            zoffset = top.zbeam#_extractvar('zbeam',varsuffix,'top',ff)
                #print("advance simulation porcentance ", 100 * i / nsteps, " %")
            #check shape of curr1 and zplmesh

            info("ppp", ppp)
            info("top.curr (curr1 ya ravel)", curr1)
            info("top.zplmesh (zplmesh)", zplmesh)
            base1="salida/"
            lineaf2 = base1 + "step_" + str(top.it) + ".png"
            lineaf1 = base1 + "histo_step_" + str(top.it) + ".png"

            plot_potential_and_current(h2_beam,zplmesh, curr1,w3d.xmmin, w3d.xmmax,w3d.zmmin, w3d.zmmax,ppp,lineaf1)

            plot_particles_3plots(h2_beam,w3d.zmmax,lineaf2) 
                
        if top.it%1 == 0:
    #             # utils.plot_rms()
    #             utils.beamplots()
    #             # print(h2_beam.getux())
                  window()
                  #limits(0, .1, 0.0, 0.02)
                  #pfxy(plotsg=0, cond=0, titles=False,iz=int(w3d.nz/2))

                  #crt1.drawxy(filled=150)
                  #pfzx(plotsg=0, cond=0, titles=False,iy=int(w3d.ny/2)-1)
                  pfzx(plotsg=0, cond=0, titles=False,iy=top.it)
                  crt1.drawzx(filled=150)
            
                  #crt1.plot()
                  #pzxedges(color='blue')
                  ppzx(titles=False)

                  #pzyedges(color='red')
                  fma()
                  refresh()

            #plt.figure()
            #plt.scatter(h2_beam.getz(), h2_beam.getx(), s=0.5, color='blue', label='H2+')
            #plt.show()

    STEP_NUM = 2000
    PARTICLE_OUTPUT_STARTSTEP = 1800
    PARTICLE_OUTPUT_FRAME_FREQ = 2

    @callfromafterstep
    def output_particles():
        if top.it > PARTICLE_OUTPUT_STARTSTEP:
            if top.it%PARTICLE_OUTPUT_FRAME_FREQ == 0:
                utils.write_hdf5_data(top.it, [h2_beam])


    starttime = time.time()
    step(STEP_NUM)
    hcp()
    endtime = time.time()
    print("Elapsed time for simulation: {} seconds".format(endtime-starttime))

    # bunch = utils.find_bunch_p(h2_beam, max_steps=1000)


if __name__ == '__main__':
    main()
