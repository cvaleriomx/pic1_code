import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
import pandas as pd
import sys
from math import cos, pi
import collections
import collections.abc
collections.Sequence = collections.abc.Sequence
from rfq_plots_library import *

plot_or_not = True
por = float(sys.argv[1:][0])
print(por * 10)
del sys.argv[1:]
params = por
import warp as wp
wp.top.nesmult = 1  # <--- IMPORTANTE para inicializar 'emltes'
#from plane_cross1_beta3 import PlaneCrossSaverFiltered   # importar tu clase

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
max_radius = 3.0 * wp.cm
time_small = 0.25e-9
wp.top.dt = time_small

wp.w3d.nx = 74
wp.w3d.ny = 74
#wp.w3d.nz = 15200
wp.w3d.nz = 1100
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
wp.w3d.xmmax = 0.03; wp.w3d.ymmax = 0.03; wp.w3d.zmmax = 0.40


wp.package("w3d")
wp.generate()
base="salida/"
filename24 =base + "merged_planes.csv"
#print(len(protons.getx()), "particulas")
XX,YY=drift(XX,VXX/VZ,YY,VYY/VZ,ZZ)
#zre2 = np.random.uniform(-z0lenght-0.001,-0.001, len(zre))

    # Warp espera arrays 1D de floats
#wp.addparticles(x=xre, y=yre, z=zre2, vx=vxre, vy=vyre, vz=vzre,js=0, lallindomain=True)

fig = plt.figure(figsize=(10, 6))
#plt.scatter(xre, vxre/vzre, s=1, label='Initial Particle Distribution')

wp.addparticles(x=XX, y=YY, z=ZZ, vx=VXX, vy=VYY, vz=VZ, js=0, lallindomain=True)
fig = plt.figure(figsize=(10, 6))
plt.scatter(XX, YY, s=1, label='Initial Particle Distribution')
plt.show()

Vrf = 80e3          # inter-vane voltage (±Vrf/2)
condid_rod = 101    # mismo ID para base y “carver” (requerido por CSG)

from math import cos,sin, pi

# Dominio y solver
#wp.w3d.xmmax = 0.02; wp.w3d.ymmax = 0.02; 
wp.w3d.zmmax = 0.40

solver = wp.MultiGrid3D()
#solver=wp.em3dsolver.EM3D()
# Parámetros RFQ
Vrf      = 80e3
r_center = 9.0e-3
R0       = 2.5e-3
dR       = 0.6e-3
Lcell    = 5.0e-3
Lz       = wp.w3d.zmmax

def rofz(z):  # perfil del radio del rod (superficie de revolución)
    return R0 + dR*cos(2*pi*z/Lcell)
def rofz2(z):  # perfil del radio del rod (superficie de revolución)
    return R0 + dR*sin(2*pi*z/Lcell)

# Cuatro rods (±x, ±y) con ±V/2
rod_xp = wp.ZSrfrvIn(rofzfunc=rofz, zmin=0.0, zmax=Lz,
                  voltage=+Vrf/2, xcent=+r_center, ycent=0.0, condid='next')
rod_xm = wp.ZSrfrvIn(rofzfunc=rofz, zmin=0.0, zmax=Lz,
                  voltage=Vrf/2, xcent=-r_center, ycent=0.0, condid='next')
rod_yp = wp.ZSrfrvIn(rofzfunc=rofz2, zmin=0.0, zmax=Lz,
                 voltage=-Vrf/2, xcent=0.0, ycent=+r_center, condid='next')
rod_ym = wp.ZSrfrvIn(rofzfunc=rofz2, zmin=0.0, zmax=Lz,
                  voltage=-Vrf/2, xcent=0.0, ycent=-r_center, condid='next')

#wp.registerconductors(rod_xp, rod_xm, rod_yp, rod_ym)
cons=rod_xp+ rod_xm+ rod_yp+ rod_ym
#cons= rod_yp+ rod_ym

wp.installconductors(cons)

#solver.mglevels = 3
#solver.generate(); 
#solver.solve()
wp.fieldsolve()
wp.window(0)
wp.fma()
wp.pfxy(plotsg=0, cond=0, titles=False)
cons.draw(filled=150, fullplane=False)
#wp.ppzr(titles=False)
wp.limits(wp.w3d.xmminglobal, 0.5*wp.w3d.xmmaxglobal,wp.w3d.xmminglobal, 0.5*wp.w3d.xmmaxglobal)
#wp.limits(-0.01, 0.01,-0.01, 0.01)

#ptitles('Hot plate source into solenoid transport', 'Z (m)', 'R (m)')
wp.refresh()

wp.window(1)
wp.fma()
wp.pzcurr()
#limits(w3d.zmminglobal, w3d.zmmaxglobal, 0., diode_current*1.5)
#wp.refresh()
#em = wp.EM3D()

if plot_or_not:
        i=0
        curr1 = wp.top.curr
        zplmesh = wp.top.zplmesh
        ppp = wp.getphi(iy=int(wp.w3d.ny/2),solver=wp.getregisteredsolver())
        #print("advance simulation porcentance ", 100 * i / nsteps, " %")
        lineaf2 = base1 + "step_" + str(i) + ".png"
        lineaf1 = base1 + "histo_step_" + str(i) + ".png"
        print(ppp.shape)
        plt.figure()
        #plor center of pp in scatter 
        poten = ppp[37, :]
        #Ex,Ey,Ez = solver.getfields()

        #wp.w3d.pfex();
        #ew3d.m.pfey(); em.pfer(); em.pfet(); em.pfez()

        #poten=ppp[:,ppp.shape[1]//2]
        zc = np.linspace(wp.w3d.zmmin, wp.w3d.zmmax, ppp.shape[1])
        print("linea ",len(zc), len(poten))
        plt.scatter(zc, poten)
        


        plot_potential_and_current(protons,zplmesh, curr1,wp.w3d.xmmin, wp.w3d.xmmax,wp.w3d.zmmin, wp.w3d.zmmax,ppp,lineaf1)
        

