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
#from warp.data_dumping import plane_save
import inspect, re
import warp.data_dumping.plane_save as ps

from warp.particles.extpart import ZCrossingParticles
from warp.particles.singleparticle import TraceParticle
from warp.diagnostics.gridcrossingdiags import GridCrossingDiags
from plane_cross1_beta3 import PlaneCrossSaverFiltered   # importar tu clase



# 1) Ver qué símbolos exporta
print([n for n in dir(ps) if not n.startswith('_')])

# 2) Listar los nombres de funciones definidas
src = inspect.getsource(ps)
print([m.group(1) for m in re.finditer(r'^def\s+(\w+)\s*\(', src, re.M)])
#print(inspect.signature(ps.ZPlane))
#print(inspect.getsource(ps.ZPlane))
print(inspect.signature(ps.PlaneSave))
#print(inspect.getsource(ps.PlaneSave))
#input()
wp.top.nesmult = 1  # <--- IMPORTANTE para inicializar 'emltes'

# Set up solenoid lattice
run_length = 0.50
drift_length = 0.02
solenoid_length = 1
solenoid_radius = 7.5e-2
NParticles = 1000
var1 = params
mag_solenoid = 0.1 * float(var1) 

#wp.top.lprntpara = False
#wp.top.lpsplots = False

# Initial BEAM variables
e_kin = 45.0 * wp.keV
emit = 10.0e-7
i_beam = -0.400 * wp.mA
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
K_perveance = (np.abs(i_beam) / I0) * factor_perveance

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

wp.top.npmax = 1*NParticles
wp.derivqty()


beam_lenght = 0.03
pulse_lenght=beam_lenght/protons.vbeam
hhh=-protons.ibeam*pulse_lenght/(wp.echarge*NParticles)
print(protons.ibeam, protons.vbeam, NParticles, hhh)
print("weight.  ////////////////////////////////////////////////////= ",hhh)
# Particle distribution 
def gauss_trunc(mu, sigma, NParticles,velocity):
    XX = np.random.normal(mu, sigma, NParticles)
    YY = np.random.normal(mu, sigma, NParticles)
    #ZZ = -0.015 + np.random.uniform(-beam_lenght/2, beam_lenght/2, NParticles)
    #dzcal=run_length/wp.w3d.nz 

    ZZ = np.random.uniform(0, 0.001, NParticles)

    #ZZ=np.zeros(NParticles)
    VXX = np.zeros(NParticles)
    VYY = np.zeros(NParticles)
    sigmavz = 0.001
    VX = np.random.normal(0, sigmavz, NParticles)
    VZ = np.random.normal(velocity, sigmavz, NParticles)
    return XX, YY, ZZ, VXX, VYY, VZ
#wp.top.sp_fract = np.array([0.025],'d') # species weight

print(wp.top.sp_fract)
#wp.top.sp_fract = wp.array([0.0],'d') # species weight
wp.top.pgroup.sw	=hhh*wp.top.sp_fract
# Conducting pipe. IT WOULD BE NICE TO HAVE THE RFQ VANES 
rfq_max_radius = 0.01
pipe = wp.ZCylinderOut(radius=rfq_max_radius, zlower=0.0, zupper=run_length)
conductors = pipe
wp.top.prwall = solenoid_radius
Use_solenoid = True
base1 = "Drift_salida/"

if Use_solenoid:
    base1 = "salida_solenoid/"
    solenoid_zi = [drift_length + i * solenoid_length + i * drift_length for i in range(3)]
    solenoid_ze = [drift_length + (i + 1) * solenoid_length + i * drift_length for i in range(3)]
    wp.addnewsolenoid(zi=solenoid_zi[0], zf=solenoid_ze[0], ri=solenoid_radius, maxbz=mag_solenoid)

sigmaz = 0.01
muz = -0.015

# --- Setup the FODO lattice
max_radius = 1.0 * wp.cm
time_small = 2.0e-9
wp.top.dt = time_small

wp.w3d.nx = 44
wp.w3d.ny = 44
#wp.w3d.nz = 15200
wp.w3d.nz = 500
dzcal=run_length/wp.w3d.nz 
wp.top.dt=dzcal/velocity
print("tiempo con haz================================== ",dzcal/velocity)
wp.w3d.xmmin = -max_radius
wp.w3d.xmmax = max_radius
wp.w3d.ymmin = -max_radius
wp.w3d.ymmax = max_radius
wp.w3d.zmmin = 0.0
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

#wp.top.inject = 2  # 2 means space-charge limited injection
#wp.top.rinject = source_curvature_radius  # Source radius of curvature
#wp.top.npinject = 150  # Approximate number of particles injected each step
#wp.w3d.l_inj_exact = True
wp.package("w3d")

#solver = wp.MultiGrid3D()
solver=wp.MRBlock3D()
wp.registersolver(solver)
xmin1 = -max_radius*0.5
xmax1 = max_radius*0.5
ymin1 = -max_radius*0.5
ymax1 = max_radius*0.5
zmin1 = run_length*0.5
zmax1 = run_length
child1 = solver.addchild(mins=[xmin1,ymin1,zmin1],
                        maxs=[xmax1,ymax1,zmax1],refinement=[2,2,2])
wp.generate()

print(len(protons.getx()), "particulas")
XX, YY, ZZ, VXX, VYY, VZ = gauss_trunc(mu, sigma, NParticles,velocity)
wp.addparticles(x=XX, y=YY, z=ZZ, vx=VXX, vy=VYY, vz=VZ, js=0, lallindomain=True)
fig = plt.figure(figsize=(10, 6))
plt.scatter(ZZ, YY, s=1, label='Initial Particle Distribution')
#plt.show()


wp.installconductors(conductors, dfill=wp.largepos)
scraper = wp.ParticleScraper(conductors)
wp.fieldsolve()

nsteps =  (run_length / velocity )/ wp.top.dt
nsteps = int(np.ceil(nsteps))*2
z0 = 0.1
#@wp.callfromafterstep
#def _myafterstep():
#    monitor = PlaneCrossSaverFiltered(
#        species=protons, z0=z0, filename="crossings_z0p10.csv",
#        xlim=(-5e-3, 5e-3), ylim=(-5e-3, 5e-3),
#        z_side="below",   # "below" = solo las que están con z < z0 al inicio
#        reseed_each_step=False  # True si hay inyección y quieres captar nuevas que entren al filtro
#    )
    
#wp.installafterstep(ping)


#wp.installafterstep(_myafterstep)

    # --- Call beamplots after every 20 steps
#wp.step(3)
#@cwp.allfromafterstep
#monitor = PlaneCrossSaverFiltered(protons, 0.1, "cross.csv")
#wp.installafterstep(monitor.step_monitor)   # aquí registras el método explícito
#wp.step(5)



z0 = 0.05
monitor = PlaneCrossSaverFiltered(
    species=protons, z0=z0, filename="crossings_z0p10.csv",
    xlim=(-2e-3, 2e-3), ylim=(-2e-3, 2e-3),
    z_side="below",           # o None / "above"
    reseed_each_step=False,
    debug=True, dz_probe=1e-2 # quítalo cuando ya funcione
)

#wp.installafterstep(monitor.step_monitor)   # <-- importante: registrar el método
#wp.step(10)


# define los dos planos
z_planes = [0.005, 0.15]
files    = ["cross_z0p001.csv", "cross_z0p300.csv"]
limits=[8e-3, 2e-3]
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

# luego corres tus pasos normalmente
#wp.step(100)
wp.window(0)

solver.drawboxzx(iy=int(wp.w3d.ny/2)    )
wp.fma()
#wp.refresh()
plt.show()
for i in range(nsteps):
    
    if False :
        wp.window(1)
        
        wp.pzcurr()
        wp.limits(wp.w3d.zmminglobal, wp.w3d.zmmaxglobal, 0.0, i_beam * 2)
        wp.fma()
        wp.refresh()

    if i % 50==0:
        curr1 = wp.top.curr
        zplmesh = wp.top.zplmesh
        curr1 = curr1.ravel()  # O también puedes usar y.reshape(-1)
        zoffset = wp.top.zbeam#_extractvar('zbeam',varsuffix,'top',ff)
        print("zoffset ", zoffset)
        ppp = wp.getphi(iy=int(wp.w3d.ny/2),solver=wp.getregisteredsolver())
        print("advance simulation porcentance ", 100 * i / nsteps, " %")
        plot_potential_and_current(protons,zplmesh, curr1,wp.w3d.xmmin, wp.w3d.xmmax,wp.w3d.zmmin, wp.w3d.zmmax,ppp)
        plot_particles_3plots(protons,run_length,i,base1)

    
    XX, YY, ZZ, VXX, VYY, VZ = gauss_trunc(mu, sigma, NParticles,velocity)
    protons.addparticles(x=XX, y=YY, z=ZZ, vx=VXX, vy=VYY, vz=VZ,js=0,lallindomain=True)
        #wp.top.npinje_s[0]=NParticles
    print("dentro del loop")
        #input()


    wp.step(1)
    #myafterstep()
    lsavephi=True
    #targetz_particles = ZCrossingParticles(zz=0.01)
    #zc = ZCrossingParticles(zz=0.1,    species=protons,    laccumulate=1)
    #x = targetz_particles.getx()
    #print("x en z=0.1 ", len(x))



    #sirve pero guarda muchisimos datos
    #ps.PlaneSave( zplane=0.1, filename="output.plk",lsavephi=False,lsaveparticles=True)
    #PlaneSave(zplane, filename=None, js=None, allways_save=0, deltaz=None, deltat=None, newfile=False, lsavephi=True, lsaveparticles=True, lsavesynchronized=False)

df = None