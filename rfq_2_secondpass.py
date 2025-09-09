import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
import pandas as pd
import sys
plot_or_not = True
por = float(sys.argv[1:][0])
print(por * 10)
del sys.argv[1:]
params = por
import warp as wp
wp.top.nesmult = 1  # <--- IMPORTANTE para inicializar 'emltes'
from plane_cross1_beta3 import PlaneCrossSaverFiltered   # importar tu clase

def drift(x,xp,y,yp,z):
    x1=x+z*xp
    y1=y+z*yp
    return x1,y1

# Set up solenoid lattice
run_length = 3.01
drift_length = 0.2
solenoid_length = 1
solenoid_radius = 7.5e-2
NParticles = 100000
var1 = params
mag_solenoid = 0.0001
0.1 * float(var1) / 10
v0 = float(var1)
wp.top.lprntpara = False
wp.top.lpsplots = False

# Initial BEAM variables
e_kin = 45.0 * wp.keV
emit = 10.0e-7
i_beam = 0.00001 * wp.mA
r_x = 1.0 * wp.mm
r_y = 1.0 * wp.mm
mu, sigma = 0, r_x

moc2 = 938.272089e6  # MeV/c^2
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
wp.top.ssnpid = wp.nextpid()

# Define ion species no used for tracking but necessary for initialization
#protons = wp.Species(type=wp.Proton, charge_state=+1, name="Protons")
protons = wp.Species(type=wp.Hydrogen, charge_state=-1, name='Hminus')

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
degrees_to_time = lambda deg, f_rf: deg / (360.0 * f_rf)
degrees_to_z = lambda deg, f_rf, v: degrees_to_time(deg, f_rf) * v
print(degrees_to_time(360, 352e6), "s en 1 ciclo a 352 MHz")
print(degrees_to_z(360, 352e6,2936039.794467285), "m en 1 ciclo a 352 MHz y v=0.0735c")
z0lenght=degrees_to_z(360, 352e6,2936039.794467285)*4

# Particle distribution 
XX = np.random.normal(mu, sigma, NParticles)
YY = np.random.normal(mu, sigma, NParticles)
ZZ = np.random.uniform(-z0lenght-0.001,-0.001, NParticles)
#ZZ = -0.015 + np.random.uniform(-0.015, 0.0149, NParticles)
VXX = np.zeros(NParticles)
VYY = np.zeros(NParticles)
VXX =np.random.uniform(-velocity*0.1, velocity*0.1, NParticles)
VYY =np.random.uniform(-velocity*0.1, velocity*0.1, NParticles)
sigmavz = 0.001
#VX = np.random.normal(0, sigmavz, NParticles)
VZ = np.random.normal(velocity, sigmavz, NParticles)
XX,YY=drift(XX,VXX/VZ,YY,VYY/VZ,ZZ)
print("velocity ", velocity)


# Conducting pipe. IT WOULD BE NICE TO HAVE THE RFQ VANES 
rfq_max_radius = 0.01
pipe = wp.ZCylinderOut(radius=rfq_max_radius, zlower=0.0, zupper=run_length)
conductors = pipe
wp.top.prwall = solenoid_radius
Use_solenoid = False
base1 = "salida/"

if Use_solenoid:
    base1 = "salida_sloneoid/"
    solenoid_zi = [drift_length + i * solenoid_length + i * drift_length for i in range(3)]
    solenoid_ze = [drift_length + (i + 1) * solenoid_length + i * drift_length for i in range(3)]
    wp.addnewsolenoid(zi=solenoid_zi[0], zf=solenoid_ze[0], ri=solenoid_radius, maxbz=mag_solenoid)

sigmaz = 0.01
muz = -0.015

# --- Setup the FODO lattice
max_radius = 1.0 * wp.cm
time_small = 0.25e-9
wp.top.dt = time_small

wp.w3d.nx = 44
wp.w3d.ny = 44
#wp.w3d.nz = 15200
wp.w3d.nz = 3100
wp.w3d.xmmin = -max_radius
wp.w3d.xmmax = max_radius
wp.w3d.ymmin = -max_radius
wp.w3d.ymmax = max_radius
wp.w3d.zmmin = -z0lenght-0.002
wp.w3d.zmmax = run_length + 0.03
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.dirichlet
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.pboundxy = wp.absorb
wp.top.fstype = 7
wp.w3d.l4symtry = False


wp.package("w3d")
wp.generate()
base="salida/"
filename24 =base + "merged_planes.csv"
#print(len(protons.getx()), "particulas")
data = np.loadtxt(filename24, delimiter=",", skiprows=1)

    # columnas que nos interesan
xre  = data[:,1]
yre  = data[:,2]
zre  = data[:,3]
vxre = data[:,4]
vyre = data[:,5]
vzre = data[:,6]
XX,YY=drift(XX,VXX/VZ,YY,VYY/VZ,ZZ)
zre2 = np.random.uniform(-z0lenght-0.001,-0.001, len(zre))

xre, yre=drift(xre,vxre/vzre,yre,vyre/vzre,zre2)
    # Warp espera arrays 1D de floats
wp.addparticles(x=xre, y=yre, z=zre2, vx=vxre, vy=vyre, vz=vzre,js=0, lallindomain=True)

fig = plt.figure(figsize=(10, 6))
plt.scatter(xre, vxre/vzre, s=1, label='Initial Particle Distribution')

#wp.addparticles(x=XX, y=YY, z=ZZ, vx=VXX, vy=VYY, vz=VZ, js=0, lallindomain=True)
fig = plt.figure(figsize=(10, 6))
plt.scatter(zre2, yre, s=1, label='Initial Particle Distribution')
plt.show()

with open('campo_corregido.pkl', 'rb') as f:
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

nt = 10000
freq_rfq = 352e6
T_rfq = 200 / freq_rfq
time_array_rfq = np.linspace(0, T_rfq, nt)
print("Tiempo de signal for RFQ", time_array_rfq[1], "s")
phase_disp_rfq = 0
data_array_cos = v0*np.cos(2 * np.pi * freq_rfq * (time_array_rfq) + phase_disp_rfq)
#fig, ax = plt.subplots()
#ax.plot(time_array_rfq, data_array_cos)
#ax.set_xlabel('Time (s)')
#ax.set_ylabel('Cosine Value')
#plt.show()

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

nsteps =  (run_length / velocity )/ wp.top.dt
nsteps = int(np.ceil(nsteps))
leng_part=[]

# define los dos planos
z_planes = [-0.0001, 3.01]
max_track=[500,None]
files    = ["cross_z0p001.csv", "cross_z0p300.csv"]
limits=[10e-3, 10e-3]
monitors = []
for z0, fname,tlim,maxt in zip(z_planes, files,limits,max_track):
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
    monitors.append(mon)
    wp.installafterstep(mon.step_monitor)   # registra cada uno
direcs=np.ones(len(zre2))
filename_initial_dist = base1 + "initial_distribution.png"
combined_array = np.vstack((protons.getx(),protons.getx(), protons.gety(), protons.getz(), protons.getvx(), protons.getvy(), protons.getvz(),protons.getpid(),direcs)).T
np.savetxt(base1 + 'initial_distribution.csv', combined_array, delimiter=',', header='t_cross,x,y,z,vx,vy,vz,pid,dir', comments='')

for i in range(nsteps):
    if i % 200 == 0:
        mass_to_ev = 1.67e-27 / 1.6e-19
        energy = 0.5 * (protons.getvx()**2 + protons.getvy()**2 + protons.getvz()**2) * mass_to_ev
        limite_E = 3.4e6
        mask = energy < limite_E

        lineaf2 = base1 + "step_" + str(i) + ".png"
        fig, axs = plt.subplots(3, 1, figsize=(10, 6))
        zip = protons.getz()
        print("Z posiciones: ", np.average(zip[mask]))
        print("advance simulation porcentance ", 100 * i / nsteps, " %")
        axs[0].scatter(zip[mask], 1000 * protons.getx()[mask], label='x')
        axs[0].scatter(zip[mask], 1000 * protons.gety()[mask], label='y')
        axs[0].set_xlim(0, run_length)
        axs[0].set_ylim(-5.0, 5.0)
        axs[0].set_xlabel('z (m)')
        axs[0].legend("upper right")
        axs[0].set_ylabel('Transverse position (mm)')
        npart1=len(protons.getx()[mask])
        leng_part.append(npart1)
        print(npart1, "particulas")

        print("Ekin max", np.max(energy[mask]), "LLLLaverage", np.average(energy[mask]))
        axs[1].hist(energy[mask], bins=100, density=True, label='Ekin', range=(42e3, 3.2e6))
        axs[1].set_xlabel('Ekin (eV)')
        axs[2].scatter(zip[mask], energy[mask], label='Ekin')
        axs[2].set_xlabel('z (m) free')
        plt.tight_layout()
        plt.savefig(lineaf2)
        plt.clf()

    wp.step()
df = None