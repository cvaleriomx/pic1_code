import matplotlib.pyplot as plt
import numpy as np

__author__ = "Daniel Winklehner v1 cristhian valerio v2 "
__doc__ = """
Warp Script with a simple solenoid lattice to demonstrate space charge compensation.

USPAS Jan 2018
"""
def calc_rms(protons):


        x1=protons.getx()
        MX=np.mean(x1)
        X=x1
        #print("media = ",MX)
        varx=0
        LS=len(X)
        for i2 in range(0,len(X)):
            varx   = varx   + (X[i2]-MX) *(X[i2]-MX)/LS
        RMS=np.sqrt(varx*varx)
        return(RMS)
def generar_muestras_uniforme(pdf, rango, num_muestras):
    muestras = []
    con=0
    while con < num_muestras:
        x = np.random.uniform(rango[0], rango[1])
        y = np.random.uniform(rango[0],rango[1])
        con=con+1
        muestras.append(x)
    #print(muestras)
    return np.array(muestras)
def pdf_uniform(n,a,b):
    y=np.random.uniform(rango[0],rango[1],20000)
    return y

def calc_emit_rms(x1,xp1):
    MX=np.mean(x1)
    MPX=np.mean(xp1)
    X=x1
    PX2=xp1
    print("media = ",MX)
    rmd=np.sqrt(np.mean(x1**2))
    print("media = ",1e3*rmd)
    LS=len(xp1)
    varx=0
    varpx=0
    varxpx=0
    for i2 in range(0,len(X)):
              	varx   = varx   + (X[i2]-MX) *(X[i2]-MX)/LS
              	varpx  = varpx  + (PX2[i2]-MPX)*(PX2[i2]-MPX)/LS
              	varxpx = varxpx + (X[i2]-MX)*(PX2[i2]-MPX)/LS
    print(varx)
    e_rms = 1*np.sqrt(varx*varpx-varxpx*varxpx)
    print ("RMS Size X = %.4f mm Emittance =  %03s mm.mrad" % (np.sqrt(varx)*1e3,e_rms*1000000))
    return (e_rms)

def plot_potential_and_current(protons,zplmesh, curr1,xmmin,xmmax,zmmin,zmmax,ppp,lineaf):
        fig4, axs2 = plt.subplots(2, 1, figsize=(10, 6))
        #plg(curr,zoffset+zplmesh/zscale,color=color,linetype=linetype,

        #axs2[0].scatter(zoffset+zplmesh, curr1, label='Current Density' )

        #axs2[1].scatter(protons.getz(), protons.getx(), label='x')
        limits = [[-0.03, 3.1], [-0.01, 0.01]]

        Hx, xedges, yedges = np.histogram2d(protons.getz(), protons.getx(), bins=(3000, 20),range=limits)
        Hx = np.rot90(Hx)
        Hx = np.flipud(Hx)
        Hmaskedx = np.ma.masked_where(Hx == 0, Hx)
        # Sumar sobre y -> distribución solo en z
        counts_z = Hmaskedx.sum(axis=0)   # suma en eje y
        z_centers = 0.5 * (xedges[:-1] + xedges[1:])  # centros de bins en z
        #axs2[0].scatter(zplmesh, curr1, label='Current Density' )
        #axs2[0].scatter(z_centers, counts_z / np.max(counts_z) * np.max(np.abs(curr1)), label='Particle Density (scaled)', color='orange', s=10)
        axs2[0].scatter(z_centers, counts_z, label='Particle Density (scaled)', color='orange', s=10)

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
        xc = np.linspace(xmmin, xmmax, ppp.shape[0])
        zc = np.linspace(zmmin, zmmax, ppp.shape[1])
        Z, X = np.meshgrid(zc, xc)  # ojo: Z primero (cols), X después (filas)
        # 4) Contornos del potencial (centros)
        levels = np.linspace(np.min(ppp), np.max(ppp), 15)
        cntr1= axs2[1].contourf(Z, X, np.ma.masked_invalid(ppp),
                      corner_mask=True, linewidths=0.5)
        axs2[1].pcolormesh(xedges, yedges, Hmaskedx, shading='auto',alpha=0.26)
        #axs2[1].scatter(protons.getz(), protons.getx(),c="red",s=0.5,alpha=0.05)
        #axs2[1].contour(Z, X, np.ma.masked_invalid(ppp), levels=14, linewidths=0.5, colors='k')
        #cntr1 = axs2[1].contourf(Z, X, np.ma.masked_invalid(ppp), levels=14, cmap="RdBu_r")
        fig4.colorbar(cntr1, ax=axs2[1], orientation='vertical')

        # 5) Heatmap de partículas (bordes/edges)
        axs2[1].set_xlabel('z')
        axs2[1].set_ylabel('x')
        
        

        plt.savefig(lineaf)

        #plt.show()


def plot_particles_3plots(protons,run_length,lineaf):
      
      
      #wp.limits(wp.w3d.zmminglobal, wp.w3d.zmmaxglobal, 0., i_beam*2)
        mass_to_ev = 1.67e-27 / 1.6e-19
        energy = 0.5 * (protons.getvx()**2 + protons.getvy()**2 + protons.getvz()**2) * mass_to_ev
        limite_E = 3.4e6
        mask = energy < limite_E

        #lineaf2 = base1 + "step_" + str(i) + ".png"
        fig, axs = plt.subplots(3, 1, figsize=(10, 6))
        zip = protons.getz()
        #print("Z posiciones: ", np.average(zip[mask]))
        
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
        #plt.show()
        plt.savefig(lineaf)
        plt.clf()

def gauss_trunc(mu, sigma, NParticles,velocity,sigmaT=0.0001,z0=0.0,z1=0.001):
    XX = np.random.normal(mu, sigma, NParticles)
    YY = np.random.normal(mu, sigma, NParticles)
    
    
    #ZZ = -0.015 + np.random.uniform(-beam_lenght/2, beam_lenght/2, NParticles)
    #dzcal=run_length/wp.w3d.nz 

    ZZ = np.random.uniform(z0, z1, NParticles)

    #ZZ=np.zeros(NParticles)
    #VXX = np.zeros(NParticles)
    #VYY = np.zeros(NParticles)
    sigmavz = 0.001
    VXX = np.random.uniform(-sigmaT*velocity, sigmaT*velocity, NParticles)
    VYY = np.random.uniform(-sigmaT*velocity, sigmaT*velocity, NParticles)
    VZ=np.sqrt(velocity-VXX*VXX-VYY*VYY)
    #VZ=np.random.normal(velocity, sigmavz, NParticles)
    #VX = np.random.normal(0, sigmavz, NParticles)
    #VZ = np.random.normal(velocity, sigmavz, NParticles)
    return XX, YY, ZZ, VXX, VYY, VZ


def  uniform_beam(mu, sigma, NParticles,velocity,sigmaVT=0.0001,z0=0.0,z1=0.001):
    XX = np.random.uniform(-sigma, sigma, NParticles)
    YY = np.random.uniform(-sigma, sigma, NParticles)
    #XX = np.rand
    
    #ZZ = -0.015 + np.random.uniform(-beam_lenght/2, beam_lenght/2, NParticles)
    #dzcal=run_length/wp.w3d.nz 

    ZZ = np.random.uniform(z0, z1, NParticles)

    #ZZ=np.zeros(NParticles)
    #VXX = np.zeros(NParticles)
    #VYY = np.zeros(NParticles)
    sigmavz = 0.001
    VXX = np.random.uniform(-sigmaVT*velocity, sigmaVT*velocity, NParticles)
    VYY = np.random.uniform(-sigmaVT*velocity, sigmaVT*velocity, NParticles)
    vxy2 = VXX**2 + VYY**2
    VZ=np.sqrt(velocity**2-vxy2)
    #VZ=np.random.normal(velocity, sigmavz, NParticles)
    #VX = np.random.normal(0, sigmavz, NParticles)
    #VZ = np.random.normal(velocity, sigmavz, NParticles)
    return XX, YY, ZZ, VXX, VYY, VZ