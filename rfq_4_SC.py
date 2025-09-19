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

# Set up solenoid lattice
run_length = 3.03
drift_length = 0.2
solenoid_length = 1
solenoid_radius = 7.5e-2
NParticles = 1000
var1 = params
mag_solenoid = 0.0001
0.1 * float(var1) / 10
v0 = float(var1)
wp.top.lprntpara = False
wp.top.lpsplots = False




# Initial BEAM variables
e_kin = 45.0 * wp.keV
emit = 10.0e-7
i_beam = 35.0000 * wp.mA
r_x = 5.0 * wp.mm
r_y = 5.0 * wp.mm
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
print("distance steps ", run_length / velocity, "s")
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


sigmaz = 0.01
muz = -0.015
time_small = 0.25e-9
time_small=3.403343e-10
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
conductors = pipe
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

wp.top.dt = time_small*2
print("distance steps ", run_length / velocity, "s")
print("time step ", wp.top.dt, "s", "which is ", wp.top.dt * velocity * 1e3, "mm")
mesh_sizez1 = velocity * wp.top.dt
wp.w3d.nx = 44
wp.w3d.ny = 44
#wp.w3d.nz = 15200
wp.w3d.nz = int((run_length) / (velocity * wp.top.dt)) + 1
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

nt = 20000
freq_rfq = 352e6
T_rfq = 400 / freq_rfq
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
#wp.solver.ldosolve = False

nsteps =  (run_length / velocity )/ wp.top.dt
nsteps = int(np.ceil(nsteps))
leng_part=[]

# define los dos planos
z_planes = [0.00, 3.0]
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
    monitors.append(mon)
    wp.installafterstep(mon.step_monitor)   # registra cada uno
def myinjection1():
            #X, YY, ZZ, VXX, VYY, VZ = gauss_trunc(mu, sigma, NParticles,velocity,z0=-2*mesh_sizez1,z1=-mesh_sizez1)
            XX, YY, ZZ, VXX, VYY, VZ = uniform_beam(mu, sigma, NParticles,velocity,0.25,z0=-2*mesh_sizez1,z1=-mesh_sizez1)
            protons.addparticles(x=XX, y=YY, z=ZZ, vx=VXX, vy=VYY, vz=VZ,js=0,lallindomain=True)
#wp.installuserinjection(myinjection1) 

XX, YY, ZZ, VXX, VYY, VZ = uniform_beam(mu, sigma, NParticles,velocity,0.25,z0=-2*mesh_sizez1,z1=-mesh_sizez1)

fig22= plt.figure()
plt.scatter(XX,YY)
plt.show()
            #XX, YY, ZZ, VXX, VYY, VZ = gauss_trunc(mu, sigma, NParticles,velocity)
protons.addparticles(x=XX, y=YY, z=ZZ, vx=VXX, vy=VYY, vz=VZ,js=0,lallindomain=True)
for i in range(nsteps):
    if i % 10==0:
        curr1 = wp.top.curr
        zplmesh = wp.top.zplmesh
        curr1 = curr1.ravel()  # O también puedes usar y.reshape(-1)
        zoffset = wp.top.zbeam#_extractvar('zbeam',varsuffix,'top',ff)
        print("zoffset ", zoffset)
        ppp = wp.getphi(iy=int(wp.w3d.ny/2),solver=wp.getregisteredsolver())
        print("advance simulation porcentance ", 100 * i / nsteps, " %")
        lineaf2 = base1 + "step_" + str(i) + ".png"
        lineaf1 = base1 + "histo_step_" + str(i) + ".png"

        plot_potential_and_current(protons,zplmesh, curr1,wp.w3d.xmmin, wp.w3d.xmmax,wp.w3d.zmmin, wp.w3d.zmmax,ppp,lineaf1)
        plot_particles_3plots(protons,run_length,lineaf2) 
    wp.step()
    XX, YY, ZZ, VXX, VYY, VZ = uniform_beam(mu, sigma, NParticles,velocity,0.25,z0=-2*mesh_sizez1,z1=-mesh_sizez1)
    protons.addparticles(x=XX, y=YY, z=ZZ, vx=VXX, vy=VYY, vz=VZ,js=0,lallindomain=True)
    
df = None