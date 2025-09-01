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
#from warp.data_dumping import plane_save
import inspect, re
import warp.data_dumping.plane_save as ps

from warp.particles.extpart import ZCrossingParticles
from warp.particles.singleparticle import TraceParticle
from warp.diagnostics.gridcrossingdiags import GridCrossingDiags
from plane_cross1_beta import PlaneCrossSaverFiltered   # importar tu clase



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
drift_length = 0.2
solenoid_length = 1
solenoid_radius = 7.5e-2
NParticles = 800
var1 = params
mag_solenoid = 0.1 * float(var1) / 10

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
#wp.top.sp_fract = np.array([0.025],'d') # species weight

print(wp.top.sp_fract)
#wp.top.sp_fract = wp.array([0.0],'d') # species weight
wp.top.pgroup.sw	=hhh*wp.top.sp_fract
# Conducting pipe. IT WOULD BE NICE TO HAVE THE RFQ VANES 
rfq_max_radius = 0.01
pipe = wp.ZCylinderOut(radius=rfq_max_radius, zlower=0.0, zupper=run_length)
conductors = pipe
wp.top.prwall = solenoid_radius
Use_solenoid = False
base1 = "Drift_salida/"

if Use_solenoid:
    base1 = "salida_sloneoid/"
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
wp.generate()

print(len(protons.getx()), "particulas")

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
def myafterstep():
    monitor = PlaneCrossSaverFiltered(
        species=protons, z0=z0, filename="crossings_z0p10.csv",
        xlim=(-5e-3, 5e-3), ylim=(-5e-3, 5e-3),
        z_side="below",   # "below" = solo las que están con z < z0 al inicio
        reseed_each_step=False  # True si hay inyección y quieres captar nuevas que entren al filtro
    )

wp.installafterstep(myafterstep)
    # --- Call beamplots after every 20 steps
#@cwp.allfromafterstep
    
 
for i in range(nsteps):
    

    if i % 50 == 0:

        wp.window(1)
        
        wp.pzcurr()
        wp.limits(wp.w3d.zmminglobal, wp.w3d.zmmaxglobal, 0.0, i_beam * 2)
        wp.fma()
        wp.refresh()

        curr1 = wp.top.curr
        zplmesh = wp.top.zplmesh
        print( curr1.shape, "current shape")
        print(min(zplmesh), max(zplmesh), "zplmesh limits")
        print(zplmesh.shape, "zplmesh shape")
        #plt.figure(figsize=(10, 6))
        fig4, axs2 = plt.subplots(2, 1, figsize=(10, 6))

        curr1 = curr1.ravel()  # O también puedes usar y.reshape(-1)
        zoffset = wp.top.zbeam#_extractvar('zbeam',varsuffix,'top',ff)
        print("zoffset ", zoffset)
        #plg(curr,zoffset+zplmesh/zscale,color=color,linetype=linetype,
        axs2[0].scatter(zoffset+zplmesh, curr1, label='Current Density' )
        axs2[0].scatter(zplmesh, curr1, label='Current Density' )

        #axs2[1].scatter(protons.getz(), protons.getx(), label='x')
        limits = [[-0.03, 1], [-0.01, 0.01]]

        Hx, xedges, yedges = np.histogram2d(protons.getz(), protons.getx(), bins=100)
        Hx = np.rot90(Hx)
        Hx = np.flipud(Hx)
        Hmaskedx = np.ma.masked_where(Hx == 0, Hx)

        ppp = wp.getphi(iy=int(wp.w3d.ny/2),solver=wp.getregisteredsolver())
                #print(ppp)
        #ppp=ppp.T
        #der=ppp.shape
        #print(der)
        #ppp = np.ma.masked_where(ppp==0,ppp)
        #levels1=np.linspace(np.min(ppp),np.max(ppp),15)
        #axs2[1].pcolormesh(xedges, yedges, Hmaskedx, shading='auto',)

        # 2) Campo (slice en y): ppp con shape (nx, nz)
        # Si viene transpuesto, corrígelo:
        if ppp.shape[0] != Hx.shape[0] and ppp.shape[1] == Hx.shape[0]:
            ppp = ppp.T  # ahora debe ser (nx, nz)

        # 3) Mallas ascendentes en x y z (centros para contourf)
        xc = np.linspace(wp.w3d.xmmin, wp.w3d.xmmax, ppp.shape[0])
        zc = np.linspace(wp.w3d.zmmin, wp.w3d.zmmax, ppp.shape[1])
        Z, X = np.meshgrid(zc, xc)  # ojo: Z primero (cols), X después (filas)
        # 4) Contornos del potencial (centros)
        levels = np.linspace(np.min(ppp), np.max(ppp), 15)
        axs2[1].contourf(Z, X, np.ma.masked_invalid(ppp),
                      corner_mask=True, linewidths=0.5)
        #axs2[1].pcolormesh(xedges, yedges, Hmaskedx, shading='auto',alpha=0.6)
        axs2[1].scatter(protons.getz(), protons.getx(),c="red",s=0.5,alpha=0.5)

        # 5) Heatmap de partículas (bordes/edges)
        axs2[1].set_xlabel('z')
        axs2[1].set_ylabel('x')
        #axs2[1].set_aspect('auto')  # o 'equal' si te conviene
        fig, ax = plt.subplots()
        # Centro geométrico de x
        x_mid = 0.5*(wp.w3d.xmmin + wp.w3d.xmmax)

        # Índice del x más cercano al centro
        ix = np.argmin(np.abs(xc - x_mid))

        # Perfil 1D: potencial vs z en x ~ mitad
        pp_line = ppp[ix, :]          # shape (nz,)
        z_line  = zc                  # mismos z que usaste para contour

        plt.figure()
        plt.plot(z_line, pp_line)
        plt.xlabel('z')
        plt.ylabel('phi (pp)')
        plt.title(f'Corte a x ≈ {xc[ix]:.3g}')
        plt.grid(True)

        #plt.xlim(0, 0.2)
        plt.show()
    if False:
        #wp.limits(wp.w3d.zmminglobal, wp.w3d.zmmaxglobal, 0., i_beam*2)
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

        print("Ekin max", np.max(energy[mask]), "LLLLaverage", np.average(energy[mask]))
        axs[1].hist(energy[mask], bins=100, density=True, label='Ekin', range=(42e3, 3.2e6))
        axs[1].set_xlabel('Ekin (eV)')
        axs[2].scatter(zip[mask], energy[mask], label='Ekin')
        axs[2].set_xlabel('z (m) free')
        plt.tight_layout()
        plt.savefig(lineaf2)
        plt.clf()
    #if i % 10==0 and i>1: 
        #wp.top.npmax = 2*NParticles

    protons.addparticles(x=XX, y=YY, z=ZZ, vx=VXX, vy=VYY, vz=VZ,js=0,lallindomain=True)
        #wp.top.npinje_s[0]=NParticles
    print("dentro del loop")
        #input()


    wp.step()
    myafterstep()
    lsavephi=True
    targetz_particles = ZCrossingParticles(zz=0.01)
    #zc = ZCrossingParticles(zz=0.1,    species=protons,    laccumulate=1)
    x = targetz_particles.getx()
    print("x en z=0.1 ", len(x))



    #sirve pero guarda muchisimos datos
    #ps.PlaneSave( zplane=0.1, filename="output.plk",lsavephi=False,lsaveparticles=True)
    #PlaneSave(zplane, filename=None, js=None, allways_save=0, deltaz=None, deltat=None, newfile=False, lsavephi=True, lsaveparticles=True, lsavesynchronized=False)

df = None