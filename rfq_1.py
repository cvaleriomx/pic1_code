from curses.panel import top_panel
import matplotlib.pyplot as plt
import numpy as np
from libreria import *
import scipy.optimize as optimize
from scipy import interpolate
#from scipy.integrate import simps
import gc
import sys

import pickle
import pandas as pd
plot_or_not = True
por = float(sys.argv[1:][0])
print(por * 10)
del sys.argv[1:]
import warp as wp
params = por
wp.top.nesmult = 1  # <--- IMPORTANTE para inicializar 'emltes'

# Set up solenoid lattice
run_length = 3.0
drift_length = 0.2
solenoid_length = 1
#0.5*run_length
solenoid_radius = 7.5e-2
NParticles = 100000
n_grid = 128
var1 = params
mag_solenoid = 0.1 * float(var1) / 10

# Prevent gist from starting upon setup
wp.top.lprntpara = False
wp.top.lpsplots = False

# Define some initial BEAM variables
e_kin = 45.0 * wp.keV
emit = 10.0e-7
i_beam = 0.00001 * wp.mA
r_x = 1.0 * wp.mm
r_y = 1.0 * wp.mm
r_xp = 0.0e-3
r_yp = r_xp
mu, sigma = 0, r_x

# Definition and derivation of kinematics
moc2 = 938.272089e6  # MeV/c^2
clight = 299792458
gammar = 1 + (e_kin) / moc2
betar = np.sqrt(1 - 1.0 / (gammar ** 2))
bg = betar * gammar
velocity = betar * clight
#print("Velocity", velocity)
#kinetic_energy = 0.5*velocity**2 * (1.67262e-27 / 1.602e-19)  # in eV
#print("Kinetic energy:", kinetic_energy, "eV")
#input()
I0 = moc2/(30) # electron rest energy in GeV
factor_perveance = 2 / (bg * bg * bg)
K_perveance = (i_beam / I0) * factor_perveance

f_pr = (1.67e-27) * 3e8 / 1.6e-19
Brho = bg * f_pr
ksol = (0.5 * mag_solenoid / Brho) ** 2
cgmplotfreq = 1000
rb0=np.sqrt(K_perveance/ksol)
print("rb0 ", rb0)
print("gg ",K_perveance, ksol,Brho,mag_solenoid, velocity, betar, gammar)
wp.top.ssnpid = wp.nextpid()
top.inject = 1

# Define ion species
protons = wp.Species(type=wp.Proton, charge_state=+1, name="Protons")
beam_species = [protons]
for beam in beam_species:
    beam.ekin = e_kin
    beam.vbeam = 0.0
    beam.ibeam = i_beam
    beam.emitx = emit
    beam.emity = emit
    beam.vthz = 0.0
    beam.a0    = r_x
    beam.b0    = r_y
    #beam.emit  = 6.247186343204832e-05
    beam.ap0   = 0.
    beam.bp0   = 0.
    beam.zbeam=0.0

wp.top.npmax = NParticles

wp.derivqty()

# Define particle distribution
XX = np.random.normal(mu, sigma, NParticles)
YY = np.random.normal(mu, sigma, NParticles)
a, c = 0.5, 20
rango = [0, int(1000 * sigma)]
#XX = muestras_circulares[:, 0]
#YY = muestras_circulares[:, 1]
ZZ = np.zeros(NParticles)
VXX = np.zeros(NParticles)
VYY = np.zeros(NParticles)
sigmavz = 0.001
VX = np.random.normal(0, sigmavz, NParticles)

VZ = np.random.normal(velocity, sigmavz, NParticles)
print("Velocidad1",velocity)
#input()
#wp.top.npmax = NParticles
#protons.ekin = e_kin

# Define the conducting pipe
pipe = wp.ZCylinderOut(radius=solenoid_radius, zlower=0.0, zupper=run_length)
#plano2 = wp.Box(voltage=0,xcent=solenoid_radius,xsize=solenoid_radius*2,ysize=0.5,zsize=0.004,ycent=0,zcent=0.4)
conductors = pipe
wp.top.prwall = solenoid_radius
realistic_solenoid = True
if realistic_solenoid:
    base1 = "salida_realistic/"
    solenoid_zi = [drift_length + i * solenoid_length + i * drift_length for i in range(3)]
    solenoid_ze = [drift_length + (i + 1) * solenoid_length + i * drift_length for i in range(3)]
    wp.addnewsolenoid(zi=solenoid_zi[0], zf=solenoid_ze[0], ri=solenoid_radius, maxbz=mag_solenoid)
else:
    base1 = "salida_ideal/"
    bzf = np.zeros([4, 4, 10])
    bzf[:, :, :] = mag_solenoid
    z_start = -0.2
    z_stop = run_length
    wp.addnewbgrd(z_start, z_stop, xs=wp.w3d.xmmin, dx=(wp.w3d.xmmax - wp.w3d.xmmin), ys=wp.w3d.ymmin, dy=(wp.w3d.ymmax - wp.w3d.ymmin), nx=2, ny=2, nz=3, bz=bzf)
sigmaz = 0.01 # standard deviation of distribution
muz=-0.015
#ZZ = muz + sigmaz * np.random.randn(NParticles)
ZZ = muz +  np.random.uniform(-0.015,0.015,NParticles)


#wp.addparticles(x=XX,y=YY,z=ZZ,vx=VXX,vy=VYY,vz=VZ,js=0,lallindomain=True)


# --- Setup the FODO lattice
# --- These are user created python variables describing the lattice.
hlp     = run_length   # half lattice period length
piperad = 1.0*wp.cm # pipe radius
quadlen = 11.0*wp.cm   # quadrupole length

# --- Magnetic quadrupole field gradient - calculated to give sigma0 = 72 degrees.
dbdx    = 0.93230106124518164/quadlen
time_small=0.5e-9
time_big=1e-9
wp.top.dt =time_small
#(top.tunelen/steps_p_perd)/beam.vbeam
print("time is ===")
print(wp.top.dt)
# --- Specify the number of grid cells in each dimension.
wp.w3d.nx = 44
wp.w3d.ny = 44
wp.w3d.nz = 15200

# --- Specify the extent of the field solve grid.
wp.w3d.xmmin = -piperad
wp.w3d.xmmax =  piperad
wp.w3d.ymmin = -piperad
wp.w3d.ymmax =  piperad
wp.w3d.zmmin = -0.03
wp.w3d.zmmax = hlp

# --- Specify the boundary conditions on the outer sides of the grid.
# --- Possible values are dirichlet, periodic, and neumann.
wp.w3d.bound0 = wp.neumann # at iz == 0
wp.w3d.boundnz = wp.neumann # at iz == nz
wp.w3d.boundxy = wp.dirichlet # at all transverse sides

# --- Set the particle boundary conditions at the outer sides of the grid.
# --- Possible values are absorb, periodic, and reflect.
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.pboundxy = wp.absorb

#wp.top.inject = 100

# --- Set up field solver.
# --- fstype == 0 species the FFT solver 7 for grid.
wp.top.fstype = 7

# --- When the charge is deposited, it would be mapped into the one quadrant.
wp.w3d.l4symtry = False

#wp.w3d.distrbtn = "semigaus"

# --- The longitudinal velocity distribution of the beam.
#wp.w3d.distr_l = "gaussian"
z_start = wp.w3d.zmmin
z_stop = wp.w3d.zmmax




#wp.derivqty()
wp.package("w3d")
wp.generate()


print(len(protons.getx()), "particulas")
if False:
    for varu in range(len(protons.getx())):
        protons.getx()[varu] = XX[varu]
        protons.gety()[varu] = YY[varu]
        protons.getvx()[varu] = VXX[varu]
        protons.getvy()[varu] = VYY[varu]
        protons.getz()[varu] = ZZ[varu]
        protons.getvz()[varu] = VZ[varu]


wp.addparticles(x=XX,y=YY,z=ZZ,vx=VXX,vy=VYY,vz=VZ,js=0,lallindomain=True)
fig = plt.figure(figsize=(10, 6))
plt.scatter(ZZ, YY, s=1, label='Initial Particle Distribution')
plt.show()
if False:
    # --- Parámetros de la cavidad RF/////
    zsc = 0.25                  # posición inicial [m]
    zec = 0.6                  # posición final [m]
    field_amplitude = 1e5     # V/m
    field_frequency = 3.4e6   # Hz
    field_period = 1.0 / field_frequency  # segundos
    phase_disp = 60*np.pi/180  # desfase en radianes
    t_max = 40.5 / field_frequency     # al menos 1.5 ciclos para interpolación segura

    # --- Función que define el campo dependiente del tiempo
    time_array = np.linspace(0.0, t_max,10000)
    data_array_sin = -field_amplitude*np.sin((time_array ) * field_frequency * 2 * np.pi + phase_disp)
    figrr= plt.figure()

    #E0 = 360e3  # V/m
    dsfield=wp.w3d.dz
    z_points = np.arange(zsc, zec + dsfield,dsfield )  # incl
    Lc = zec - zsc
    fcv = velocity / (2e6 * Lc)  # frecuencia de la onda
    print("Frecuencia de la onda :", fcv, "MHz",180*2* fcv*1e6*zsc/velocity)
    es1 =1 * np.sin(np.pi * (z_points - zsc) / Lc)
    ztestarray = np.linspace(0, run_length,1000)
    estest =1 * np.sin(np.pi * (ztestarray - zsc) / Lc)
    plt.plot(ztestarray,estest, label='Campo RF (sin)')
    plt.plot(z_points, es1, label='Campo RF (es1)')
    nn = np.array([0.0, 0.0, 1.0])  # campo en z
    vv = np.array([1.0, 0.0, 0.0])  # vector transversal

    # Dummy field values (2 puntos)
    es_dummy = np.array([1.0, 1.0])
    es_1 = np.array([1.0, 1.0,1,1,1,1,1])
    # Crear EMLT
    elm = wp.addnewemlt(zs=zsc, ze=zec, es=es1,time=time_array,data=data_array_sin)
#plt.plot(time_array, data_array_sin)



with open('campo_corregido.pkl', 'rb') as f:
    df = pickle.load(f)  # o usa np.load si no es un DataFrame

# Si es un DataFrame:
x = df['x'].values
y = df['y'].values
z = df['z'].values
df['x'] *= 0.01
df['y'] *= 0.01
df['z'] *= 0.01
ex = df['Ex'].values
ey = df['Ey'].values
ez = df['Ez'].values
nx_rfq = df['x'].nunique()
ny_rfq = df['y'].nunique()
nz_rfq = df['z'].nunique()
# Ordenar el DataFrame por coordenadas (importante)
df = df.sort_values(by=['x', 'y', 'z'])

# Extraer y reshape (asumo orden C: z más rápido)
ex_rfq = df['Ex'].values.reshape((nx_rfq, ny_rfq, nz_rfq), order='C')
ey_rfq = df['Ey'].values.reshape((nx_rfq, ny_rfq, nz_rfq), order='C')
ez_rfq = df['Ez'].values.reshape((nx_rfq, ny_rfq, nz_rfq), order='C')
# Ordenar el DataFrame por coordenadas (importante)
# Número de pasos de tiempo
nt = 10000

# Frecuencia
freq_rfq = 352e6  # Hz
# Vector de tiempo cubriendo un periodo
T_rfq = 200 / freq_rfq
time_array_rfq = np.linspace(0, T_rfq, nt)
# Fase inicial (por ejemplo 0 o pi/2)
phase_disp_rfq = 0
# Modulación temporal del campo
data_array_cos = np.cos(2 * np.pi * freq_rfq * (time_array_rfq) + phase_disp_rfq)
xs_rfq = df['x'].min()
xe_rfq = df['x'].max()
ys_rfq = df['y'].min()
ye_rfq = df['y'].max()
dx_rfq = (df['x'].max() - df['x'].min()) / (nx_rfq - 1)
dy_rfq = (df['y'].max() - df['y'].min()) / (ny_rfq - 1)
dz_rfq = (df['z'].max() - df['z'].min()) / (nz_rfq - 1)
zs_rfq = df['z'].min()
ze_rfq = df['z'].max()
print("parametros RFQ",dx_rfq, dy_rfq, zs_rfq, ze_rfq,dz_rfq)
print("nx_rfq, ny_rfq, nz_rfq", nx_rfq, ny_rfq, nz_rfq)
print("xs_rfq, xe_rfq, ys_rfq, ye_rfq", xs_rfq, xe_rfq, ys_rfq, ye_rfq)
#print("time_array_r fq", time_array_rfq)
#input()
wp.addnewegrd(
    zs=zs_rfq, ze=ze_rfq,        # límites z de la región de campo
    dx=dx_rfq, dy=dy_rfq,        # tamaño de celda en x e y
    xs=xs_rfq, ys=ys_rfq,        # origen de la malla (xmin, ymin)
    time=time_array_rfq,     # vector temporal
    data=data_array_cos, # amplitud temporal (por ejemplo cosenoidal)
    ex=ex_rfq, ey=ey_rfq, ez=ez_rfq)
    #wp.addnewbgrd(z_start, z_stop, xs=wp.w3d.xmmin, dx=(wp.w3d.xmmax - wp.w3d.xmmin), ys=wp.w3d.ymmin, dy=(wp.w3d.ymmax - wp.w3d.ymmin), nx=2, ny=2, nz=3, bz=bzf)


#plt.show()
# --- Crear elemento de campo eléctrico tipo multipolo (EMLT)
#elm = wp.addnewemlt(zs=0, ze=1.1,time=time_array, func=data_array_sin)  # se sobreescribirá luego con función
#es1 = np.array([1.0, 1.0])
#elm = wp.addnewemlt(zs=0.1, ze=1.0, es=es1)
# Dirección del campo

#elm = wp.addnewemlt(zs=zs, ze=ze, es=es_dummy, nn=nn, vv=vv)



#wp.TimeDependentLatticeElement('emltes', elm, func=get_rf_field)

# --- Asociar dependencia temporal usando la función
#wp.TimeDependentLatticeElement('emltex', elm, func=get_rf_field)

wp.installconductors(conductors, dfill=wp.largepos)
#wp.installparticlescraper(conductors)
scraper = wp.ParticleScraper(conductors)


wp.fieldsolve()

nsteps =int(np.ceil(run_length / wp.w3d.dz))
guardar = 2
nsteps2 = int(nsteps / guardar)

z_posi, x_rms, x_emit, rhoprome, Npart_time, chi, phasead, trans, p1x, p1y, p1r, p2r, p3r, p4r, p5r, p1vx, p1vy, p1vr, p2vr, p3vr, p4vr, p5vr, p1x, p1xp, p1b, x_rms2, z_posi2, \
rbeam, vrbeam, vrbeam_std, p5x, p5vx, p5vy, p5y ,p1ex,p1ey,p1ez= [[0] * nsteps2 for _ in range(37)]

by222 = wp.getb(comp='B', iz=0, fullplane=1, local=0)
print("particulas",len(protons.getx()))
base1 = "salida/"
for i in range(nsteps):
    
    if i % 5 == 0:
        mass_to_ev= 1.67e-27 / 1.6e-19
        
        energy=0.5*(protons.getvx()**2 + protons.getvy()**2 + protons.getvz()**2)*mass_to_ev
        # Límite para E
        limite_E = 3.4e6
        print("N particulas ",len(energy))
        # Filtrar los puntos
        mask = energy < limite_E 

        lineaf2 = base1 + "step_" + str(i) + ".png"
        fig, axs = plt.subplots(3, 1, figsize=(10, 6))
        zip=protons.getz()
        print("Z posiciones: ", np.average(zip[mask]))
        print("advance simulation porcentance ", 100*i/nsteps, " %")
        print(protons.getx())
        axs[0].scatter(zip[mask], 1000*protons.getx()[mask], label='x')
        axs[0].scatter(zip[mask], 1000*protons.gety()[mask], label='y')
        axs[0].set_xlim(0, run_length)
        axs[0].set_ylim(-5.0, 5.0)
        axs[0].set_xlabel('z (m)')
        axs[0].set_ylabel('Transverse position (mm)')


        print("Ekin max", np.max(energy[mask]), "LLLLaverage", np.average(energy[mask]))
        axs[1].hist(energy[mask], bins=100, density=True, label='Ekin',range=(42e3, 3.2e6))
        axs[1].set_xlabel('Ekin (eV)')
        axs[2].scatter(zip[mask], energy[mask], label='Ekin')
        axs[2].set_xlabel('z (m) free')
        #axs[2].set_ylim(40e3, 55e3)
        plt.tight_layout()
        #plt.show()
        plt.savefig(lineaf2)
        plt.clf()

    wp.step()
    if False:
            plt.figure(figsize=(10, 6))
            iz = nz_rfq // 2  # plano en el centro de la cavidad

            # Extraer Ez en ese plano
            Ez_plane = ex_rfq[:, :, iz]  # plano en z = z[iz]
            # Graficar Ez en el plano
            plt.imshow(Ez_plane, extent=(xs_rfq, xe_rfq, ys_rfq, ye_rfq), origin='lower', aspect='auto', cmap='viridis',vmax=1e8)
            plt.colorbar(label='Ez (V/m)')



            iy = ny_rfq // 2
            Ez_xz = ez_rfq[:, iy, :]

            x_vals = np.sort(df['x'].unique())
            z_vals = np.sort(df['z'].unique())
            X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')

            fig55=plt.figure()
            print(np.min(Ez_xz), np.max(Ez_xz))
            vmin = np.min(Ez_xz)
            vmax = -np.min(Ez_xz)
            levelsc = np.linspace(vmin, vmax, 100)  # 100 niveles entre vmin y vmax

            plt.contourf(X, Z, Ez_xz, levels=levelsc, cmap='plasma')
            plt.colorbar(label='Ez (V/m)')
            plt.xlabel('x (m)')
            plt.ylabel('z (m)')
            #plt.title(f'Ez at y = {df_pkl["y"].unique()[iy]:.3f} m')
            #plt.axis('equal')
            plt.tight_layout()
            plt.show()
            
    df = None  # o df = pd.DataFrame() si quieres que siga siendo un DataFrame

           
    
    