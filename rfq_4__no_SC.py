import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
import pandas as pd
import sys
from rfq_plots_library import *

plot_or_not = True
por = float(sys.argv[1:][0])
print(por * 10)
del sys.argv[1:]
params = por
import warp as wp
wp.top.nesmult = 1  # <--- IMPORTANTE para inicializar 'emltes'
from plane_cross1_beta3 import PlaneCrossSaverFiltered   # importar tu clase
from rfq_vane_builder import generar_vanes

# Set up solenoid lattice
run_length = 2.13
#run_length = 1.03
drift_length = 0.2
solenoid_length = 1
solenoid_radius = 7.5e-2
NParticles = 1000
var1 = params
mag_solenoid = 0.0001
v0 = float(var1)
# Aqui colocamos el valor de v0 que queremos usar para la señal RFQ. Este valor se usará para escalar la señal cosenoidal que se aplicará a los campos eléctricos del RFQ. Puedes ajustar este valor según tus necesidades para obtener la amplitud deseada de la señal RFQ.
v0=50.2e3*float(var1)
v0=35.2e3*float(var1)
wp.top.lprntpara = False
wp.top.lpsplots = False




# Initial BEAM variables
e_kin = 28.84*wp.keV
e_kin = 30.0*wp.keV

emit = 10.0e-7
i_beam = 0.000000001000 * wp.mA
r_x = 0.50 * wp.mm
r_y = 0.50 * wp.mm
mu, sigma = 0, r_x

moc2 = 938.272089e6  # MeV/c^2
#moc2 =  13041.02e6 # MeV/c^2
clight = 299792458
gammar = 1 + (e_kin) / moc2
betar = np.sqrt(1 - 1.0 / (gammar ** 2))
bg = betar * gammar
velocity = betar * clight
I0 = moc2/(30)
factor_perveance = 2 / (bg * bg * bg)
K_perveance = (i_beam / I0) * factor_perveance

f_pr = (1.67e-27) * 3e8 / 1.6e-19
Brho = bg * f_pr
ksol = (0.5 * mag_solenoid / Brho) ** 2
rb0 = np.sqrt(K_perveance/ksol)
print("rb0 ", rb0)
print("gg ", K_perveance, ksol, Brho, mag_solenoid, velocity, betar, gammar)
print("distance steps ", run_length / velocity, "s")
wp.top.ssnpid = wp.nextpid()
#wp.top.tbirthpid=1

# Define ion species no used for tracking but necessary for initialization
protons = wp.Species(type=wp.Proton, charge_state=+1, name="Protons")
#protons = wp.Species(type=wp.Hydrogen, charge_state=-1, name='Hminus')

beam_species = [protons]
for beam in beam_species:
    beam.ekin = e_kin
    beam.ibeam = i_beam
    beam.emitx = emit
    beam.emity = emit
    beam.a0 = r_x
    beam.b0 = r_y
    beam.ap0 = 0.
    beam.bp0 = 0.
    beam.zbeam = 0.0

wp.top.npmax = NParticles
wp.derivqty()


sigmaz = 0.01
muz = -0.015
time_small = 0.25e-9
time_small=2.0e-9
freq_rfq = 35.435e6
freq_rfq = 162.0e6

T_rfq = 1 / (freq_rfq*150)
time_small = T_rfq 

Nzsteps =run_length/(velocity*time_small)+1
print("Nzsteps ", Nzsteps,"mesh sizez ",velocity*time_small)



#XX, YY, ZZ, VXX, VYY, VZ = gauss_trunc(mu, sigma, NParticles,velocity,z0=-0.001,mesh_sizez=velocity*time_small)

#ZZ = -0.015 + np.random.uniform(-0.015, 0.0149, NParticles)
#VXX = np.zeros(NParticles)
#VYY = np.zeros(NParticles)
sigmavz = 0.001
VX = np.random.normal(0, sigmavz, NParticles)
VZ = np.random.normal(velocity, sigmavz, NParticles)

# Conducting pipe. IT WOULD BE NICE TO HAVE THE RFQ VANES 
rfq_max_radius = 0.01
pipe = wp.ZCylinderOut(radius=rfq_max_radius, zlower=0.0, zupper=run_length)
vane_table_file = "../RM2/olivier_1/parmteqoutput.txt"
vane_table_file = "table1.txt"

vane_radius = 0.001
vane_conductors = generar_vanes(
    vane_table_file,
    sim_start=0.0,
    sim_end=run_length,
    sim_radius=rfq_max_radius,
    vane_radius=vane_radius,
    pts_per_cell=10,
    unit_scale=0.01,
    voltage=100.0,
)
#conductors = pipe #+
conductors =vane_conductors
wp.top.prwall = solenoid_radius
Use_solenoid = False
base1 = "salida/"

if Use_solenoid:
    base1 = "salida_sloneoid/"
    solenoid_zi = [drift_length + i * solenoid_length + i * drift_length for i in range(3)]
    solenoid_ze = [drift_length + (i + 1) * solenoid_length + i * drift_length for i in range(3)]
    #wp.addnewsolenoid(zi=solenoid_zi[0], zf=solenoid_ze[0], ri=solenoid_radius, maxbz=mag_solenoid)


#Nzsteps = 250
beam_lenght = run_length/Nzsteps*1.0
pulse_lenght=beam_lenght/protons.vbeam
hhh=protons.ibeam*pulse_lenght/(wp.echarge*NParticles)


print(wp.top.sp_fract)
#wp.top.sp_fract = wp.array([0.0],'d') # species weight
wp.top.pgroup.sw	=hhh*wp.top.sp_fract


# --- Setup the FODO lattice
max_radius = 1.0 * wp.cm

wp.top.dt = time_small*1.0
print("distance steps ", run_length / velocity, "s")
print("time step ", wp.top.dt, "s", "which is ", wp.top.dt * velocity * 1e3, "mm")
mesh_sizez1 = velocity * wp.top.dt
wp.w3d.nx = 44
wp.w3d.ny = 44
wp.w3d.nz = 200#int(run_length/mesh_sizez1)
#wp.w3d.nz = Nzsteps 
wp.w3d.xmmin = -max_radius
wp.w3d.xmmax = max_radius
wp.w3d.ymmin = -max_radius
wp.w3d.ymmax = max_radius
wp.w3d.zmmin = -2*mesh_sizez1
wp.w3d.zmmax = run_length
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.dirichlet
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.pboundxy = wp.absorb
wp.top.fstype = 7
wp.w3d.l4symtry = False
wp.top.inject = 1


wp.package("w3d")
wp.generate()

print(len(protons.getx()), "particulas")

with open('campo_corregido_ISAC2.pkl', 'rb') as f:
    df = pickle.load(f)

df['x'] *= 0.01
df['y'] *= 0.01
df['z'] *= 0.01
ex = df['Ex'].values
ey = df['Ey'].values
ez = df['Ez'].values
nx_rfq = df['x'].nunique()
ny_rfq = df['y'].nunique()
nz_rfq = df['z'].nunique()
df = df.sort_values(by=['x', 'y', 'z'])

ex_rfq = df['Ex'].values.reshape((nx_rfq, ny_rfq, nz_rfq), order='C')
ey_rfq = df['Ey'].values.reshape((nx_rfq, ny_rfq, nz_rfq), order='C')
ez_rfq = df['Ez'].values.reshape((nx_rfq, ny_rfq, nz_rfq), order='C')

nt = 2400
simulation_dt = time_small * 10.0

# La tabla temporal debe cubrir los nt pasos con el dt usado al avanzar.
time_array_rfq = np.arange(nt + 1) * simulation_dt
print("Tiempo de signal for RFQ", time_array_rfq[1], "s")
phase_disp_rfq = 0
data_array_cos = v0*np.cos(2 * np.pi * freq_rfq * (time_array_rfq) + phase_disp_rfq)
fig, ax = plt.subplots()
#ax.scatter(time_array_rfq/wp.top.dt, data_array_cos)
ax.plot(time_array_rfq,data_array_cos)
ax.scatter(time_array_rfq, data_array_cos)


ax.set_xlabel('Time (s)')
ax.set_ylabel('Cosine Value')
#plot ez along z for x=y=0
ez_along_z = ez_rfq[nx_rfq//2, ny_rfq//2, :]
plt.figure()
plt.plot(np.linspace(df['z'].min(), df['z'].max(), nz_rfq), ez_along_z)
plt.xlabel('z (m)')
plt.ylabel('Ez (V/m)')
plt.title('Ez along z-axis at x=y=0')
plt.grid()


#plot potential along z for x=y=0
potential_along_z = ez_along_z * (df['z'].max() - df['z'].min()) / nz_rfq
plt.figure()
plt.plot(np.linspace(df['z'].min(), df['z'].max(), nz_rfq), potential_along_z)
plt.xlabel('z (m)')
plt.ylabel('Potential (V)')
plt.show()
xs_rfq = df['x'].min()
xe_rfq = df['x'].max()
ys_rfq = df['y'].min()
ye_rfq = df['y'].max()
dx_rfq = (df['x'].max() - df['x'].min()) / (nx_rfq - 1)
dy_rfq = (df['y'].max() - df['y'].min()) / (ny_rfq - 1)
dz_rfq = (df['z'].max() - df['z'].min()) / (nz_rfq - 1)
zs_rfq = df['z'].min()
ze_rfq = df['z'].max()
print("parametros RFQ", dx_rfq, dy_rfq, zs_rfq, ze_rfq, dz_rfq)
print("nx_rfq, ny_rfq, nz_rfq", nx_rfq, ny_rfq, nz_rfq)
print("xs_rfq, xe_rfq, ys_rfq, ye_rfq", xs_rfq, xe_rfq, ys_rfq, ye_rfq)

wp.addnewegrd(
    zs=zs_rfq, ze=ze_rfq,
    dx=dx_rfq, dy=dy_rfq,
    xs=xs_rfq, ys=ys_rfq,
    time=time_array_rfq,
    data=data_array_cos,
    ex=ex_rfq, ey=ey_rfq, ez=ez_rfq
)

wp.installconductors(conductors, dfill=wp.largepos)
scraper = wp.ParticleScraper(conductors)
wp.fieldsolve()
#wp.solver.ldosolve = False
wp.top.dt = simulation_dt


def integrate_from_reference(values, spacing, reference_index, axis=0):
    """Integra values desde reference_index a lo largo del eje indicado."""
    values = np.moveaxis(np.asarray(values), axis, 0)
    integral = np.zeros_like(values, dtype=float)

    right_segments = 0.5 * (values[reference_index:-1] + values[reference_index + 1:]) * spacing
    if right_segments.size:
        integral[reference_index + 1:] = np.cumsum(right_segments, axis=0)

    left_segments = 0.5 * (values[:reference_index] + values[1:reference_index + 1]) * spacing
    if left_segments.size:
        integral[:reference_index] = -np.flip(
            np.cumsum(np.flip(left_segments, axis=0), axis=0),
            axis=0,
        )

    return np.moveaxis(integral, 0, axis)


def plot_rfq_potential_after_first_step_field_loaded():
    """Reconstruye y guarda el potencial del campo RFQ aplicado."""
    ix0 = nx_rfq // 2
    iy0 = ny_rfq // 2

    x_values = np.linspace(xs_rfq, xe_rfq, nx_rfq)
    y_values = np.linspace(ys_rfq, ye_rfq, ny_rfq)
    z_values = np.linspace(zs_rfq, ze_rfq, nz_rfq)

    # Ex, Ey y Ez estan normalizados en V/m por voltio intervane.
    voltage_now = v0 * np.cos(2.0 * np.pi * freq_rfq * wp.top.time + phase_disp_rfq)

    # Potencial sobre el eje, tomando Phi(z=zs_rfq)=0 como referencia.
    phi_axis = -integrate_from_reference(
        ez_rfq[ix0, iy0, :],
        dz_rfq,
        reference_index=0,
    )

    # Planos longitudinales y=0 (ZX) y x=0 (ZY).
    phi_zx = (
        phi_axis[np.newaxis, :]
        - integrate_from_reference(ex_rfq[:, iy0, :], dx_rfq, ix0, axis=0)
    ) * voltage_now
    phi_zy = (
        phi_axis[np.newaxis, :]
        - integrate_from_reference(ey_rfq[ix0, :, :], dy_rfq, iy0, axis=0)
    ) * voltage_now

    z_fractions = [0.10, 0.35, 0.65, 0.90]
    z_indices = [
        int(round(fraction * (nz_rfq - 1)))
        for fraction in z_fractions
    ]

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    for axis, iz in zip(axes.flat[:4], z_indices):
        phi_x_centerline = (
            phi_axis[iz]
            - integrate_from_reference(ex_rfq[:, iy0, iz], dx_rfq, ix0)
        )
        phi_xy = (
            phi_x_centerline[:, np.newaxis]
            - integrate_from_reference(ey_rfq[:, :, iz], dy_rfq, iy0, axis=1)
        ) * voltage_now

        image = axis.pcolormesh(
            x_values * 1e3,
            y_values * 1e3,
            phi_xy.T,
            shading="auto",
            cmap="RdBu_r",
        )
        contours = axis.contour(
            x_values * 1e3,
            y_values * 1e3,
            phi_xy.T,
            levels=15,
            colors="black",
            linewidths=0.45,
            alpha=0.75,
        )
        axis.clabel(contours, inline=True, fontsize=6, fmt="%.2g")
        axis.set_aspect("equal")
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        axis.set_title(f"Plano XY, z = {z_values[iz]:.3f} m")
        figure.colorbar(image, ax=axis, label="Potencial [V]")

    image_zx = axes[1, 1].pcolormesh(
        z_values,
        x_values * 1e3,
        phi_zx,
        shading="auto",
        cmap="RdBu_r",
    )
    contours_zx = axes[1, 1].contour(
        z_values,
        x_values * 1e3,
        phi_zx,
        levels=20,
        colors="black",
        linewidths=0.4,
        alpha=0.7,
    )
    axes[1, 1].clabel(contours_zx, inline=True, fontsize=6, fmt="%.2g")
    axes[1, 1].set_xlabel("z [m]")
    axes[1, 1].set_ylabel("x [mm]")
    axes[1, 1].set_title("Plano longitudinal ZX, y = 0")
    figure.colorbar(image_zx, ax=axes[1, 1], label="Potencial [V]")

    image_zy = axes[1, 2].pcolormesh(
        z_values,
        y_values * 1e3,
        phi_zy,
        shading="auto",
        cmap="RdBu_r",
    )
    contours_zy = axes[1, 2].contour(
        z_values,
        y_values * 1e3,
        phi_zy,
        levels=20,
        colors="black",
        linewidths=0.4,
        alpha=0.7,
    )
    axes[1, 2].clabel(contours_zy, inline=True, fontsize=6, fmt="%.2g")
    axes[1, 2].set_xlabel("z [m]")
    axes[1, 2].set_ylabel("y [mm]")
    axes[1, 2].set_title("Plano longitudinal ZY, x = 0")
    figure.colorbar(image_zy, ax=axes[1, 2], label="Potencial [V]")

    figure.suptitle(
        f"Potencial RFQ despues del paso 1, t={wp.top.time:.4e} s, "
        f"V={voltage_now:.4e} V"
    )
    output_path = base1 + "rfq_potential_after_step_1.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print("RFQ potential diagnostic saved in", output_path)
def plot_rfq_potential_after_first_step_field_warp_internal():
    """Grafica el potencial calculado internamente por Warp."""
    solver = wp.getregisteredsolver()
    x_values = np.linspace(wp.w3d.xmmin, wp.w3d.xmmax, wp.w3d.nx + 1)
    y_values = np.linspace(wp.w3d.ymmin, wp.w3d.ymmax, wp.w3d.ny + 1)
    z_values = np.linspace(wp.w3d.zmmin, wp.w3d.zmmax, wp.w3d.nz + 1)

    z_fractions = [0.10, 0.35, 0.65, 0.90]
    z_indices = [
        int(round(fraction * wp.w3d.nz))
        for fraction in z_fractions
    ]
    ix0 = int(wp.w3d.nx // 2)
    iy0 = int(wp.w3d.ny // 2)

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    # wp.getphi(iz=...) devuelve el plano XY para la posicion z seleccionada.
    for axis, iz in zip(axes.flat[:4], z_indices):
        phi_xy = np.asarray(wp.getphi(iz=iz, solver=solver))
        image = axis.pcolormesh(
            x_values * 1e3,
            y_values * 1e3,
            phi_xy.T,
            shading="auto",
            cmap="RdBu_r",
        )
        contours = axis.contour(
            x_values * 1e3,
            y_values * 1e3,
            phi_xy.T,
            levels=15,
            colors="black",
            linewidths=0.45,
            alpha=0.75,
        )
        axis.clabel(contours, inline=True, fontsize=6, fmt="%.2g")
        axis.set_aspect("equal")
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        axis.set_title(f"Plano XY, z = {z_values[iz]:.3f} m")
        figure.colorbar(image, ax=axis, label="Potencial [V]")

    # wp.getphi(iy=...) devuelve el plano longitudinal XZ.
    phi_xz = np.asarray(wp.getphi(iy=iy0, solver=solver))
    image_xz = axes[1, 1].pcolormesh(
        z_values,
        x_values * 1e3,
        phi_xz,
        shading="auto",
        cmap="RdBu_r",
    )
    contours_xz = axes[1, 1].contour(
        z_values,
        x_values * 1e3,
        phi_xz,
        levels=20,
        colors="black",
        linewidths=0.4,
        alpha=0.7,
    )
    axes[1, 1].clabel(contours_xz, inline=True, fontsize=6, fmt="%.2g")
    axes[1, 1].set_xlabel("z [m]")
    axes[1, 1].set_ylabel("x [mm]")
    axes[1, 1].set_title("Plano longitudinal XZ, y = 0")
    figure.colorbar(image_xz, ax=axes[1, 1], label="Potencial [V]")

    # wp.getphi(ix=...) devuelve el plano longitudinal YZ.
    phi_yz = np.asarray(wp.getphi(ix=ix0, solver=solver))
    image_yz = axes[1, 2].pcolormesh(
        z_values,
        y_values * 1e3,
        phi_yz,
        shading="auto",
        cmap="RdBu_r",
    )
    contours_yz = axes[1, 2].contour(
        z_values,
        y_values * 1e3,
        phi_yz,
        levels=20,
        colors="black",
        linewidths=0.4,
        alpha=0.7,
    )
    axes[1, 2].clabel(contours_yz, inline=True, fontsize=6, fmt="%.2g")
    axes[1, 2].set_xlabel("z [m]")
    axes[1, 2].set_ylabel("y [mm]")
    axes[1, 2].set_title("Plano longitudinal YZ, x = 0")
    figure.colorbar(image_yz, ax=axes[1, 2], label="Potencial [V]")

    figure.suptitle(
        f"Potencial interno de Warp despues del paso 1, "
        f"t={wp.top.time:.4e} s"
    )
    output_path = base1 + "rfq_potential_warp_internal_after_step_1.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print("Warp internal potential diagnostic saved in", output_path)


nsteps =  10*(run_length / velocity )/ wp.top.dt
print(nsteps)
#input()
nsteps = int(np.ceil(nsteps))
leng_part=[]

# define los dos planos
z_planes = [0.001, 2.0]
files    = ["cross_z0p001.csv", "cross_z0p300.csv"]
limits=[20e-3, 10e-3]
monitors = []
for z0, fname,tlim in zip(z_planes, files,limits):
    mon = PlaneCrossSaverFiltered(
        species=protons,
        z0=z0,
        filename=fname,
        xlim=(-tlim, tlim),    # opcional
        ylim=(-tlim, tlim),    # opcional
        z_side=None,           # o "below"/"above" según necesites
        reseed_each_step=True,
        include_dir=True,
        debug=False           # pon False para producción
    )
    #monitors.append(mon)
    #wp.installafterstep(mon.step_monitor)   # registra cada uno
def myinjection1():
            #coordinate generation for the beam injection using time step and velocity to determine the z position of the injected particles for z
            #where z is between the previus step and the current step avance of the particles i z
            time44 = wp.top.time
            zl_injection = velocity * time44
            #print("z_injection ", z_injection)

            #X, YY, ZZ, VXX, VYY, VZ = gauss_trunc(mu, sigma, NParticles,velocity,z0=-2*mesh_sizez1,z1=-mesh_sizez1)
            XX, YY, ZZ, VXX, VYY, VZ = uniform_beam(mu, sigma, NParticles,velocity,0.00025,z0=-2*mesh_sizez1,z1=-mesh_sizez1)
            protons.addparticles(x=XX, y=YY, z=ZZ, vx=VXX, vy=VYY, vz=VZ,js=0,lallindomain=True)
#wp.installuserinjection(myinjection1) 

XX, YY, ZZ, VXX, VYY, VZ = uniform_beam(mu, sigma, NParticles,velocity,0.25,z0=-2*mesh_sizez1,z1=-mesh_sizez1)

fig22= plt.figure()
plt.scatter(XX,YY)
plt.show()
            #XX, YY, ZZ, VXX, VYY, VZ = gauss_trunc(mu, sigma, NParticles,velocity)
protons.addparticles(x=XX, y=YY, z=ZZ, vx=VXX, vy=VYY, vz=VZ,js=0,lallindomain=True)
#for i in range(nsteps):
for i in range(nt):
    #time44=protons.getdt()
    #time44 = wp.top.time-wp.getpid(id=wp.top.tbirthpid-1, js=0)
    time44 = protons.getx()*0.0    +   (wp.top.time)
    if i ==  1000:
         #save particle data to text file 
         # cols = ["t_cross","x","y","z","vx","vy","vz","pid"]
        time44 = protons.getx()*0.0    +   (wp.top.time)
        data = {
            't_cross': time44,
            'x': protons.getx(),
            'y': protons.gety(),
            'z': protons.getz(),
            'vx': protons.getvx(),
            'vy': protons.getvy(),
            'vz': protons.getvz(),
            'pid': protons.getpid()}
        df_particles = pd.DataFrame(data)
        df_particles.to_csv('particle_data_step1000.csv', index=False)
    if i % 20==0:
        curr1 = wp.top.curr
        zplmesh = wp.top.zplmesh
        curr1 = curr1.ravel()  # O también puedes usar y.reshape(-1)
        zoffset = wp.top.zbeam#_extractvar('zbeam',varsuffix,'top',ff)
        print("zoffset ", zoffset)
        ppp = wp.getphi(iy=int(wp.w3d.ny/2),solver=wp.getregisteredsolver())
        print("advance simulation porcentance ", 100 * i / nsteps, " %")
        lineaf2 = base1 + "step_" + str(i) + ".png"
        lineaf1 = base1 + "histo_step_" + str(i) + ".png"

        plot_potential_and_current(protons,zplmesh, zplmesh,wp.w3d.xmmin, wp.w3d.xmmax,wp.w3d.zmmin, wp.w3d.zmmax,ppp,lineaf1)
        plot_particles_3plots(protons,run_length,lineaf2) 
    wp.step()
    if i == 0:
        plot_rfq_potential_after_first_step_field_loaded()
        plot_rfq_potential_after_first_step_field_warp_internal()

    if i< 9000:
        XX, YY, ZZ, VXX, VYY, VZ = uniform_beam(mu, sigma, NParticles,velocity,0.25,z0=-2*mesh_sizez1,z1=-mesh_sizez1)
        protons.addparticles(x=XX, y=YY, z=ZZ, vx=VXX, vy=VYY, vz=VZ,js=0,lallindomain=True)
        
df = None
