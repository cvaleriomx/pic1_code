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
        limits = [[zmmin, zmmax], [-0.01, 0.01]]


        Hx, xedges, yedges = np.histogram2d(protons.getz(), protons.getx(), bins=(3000, 20),range=limits)
        Hx = np.rot90(Hx)
        Hx = np.flipud(Hx)
        Hmaskedx = np.ma.masked_where(Hx == 0, Hx)
        # Sumar sobre y -> distribución solo en z
        counts_z = Hmaskedx.sum(axis=0)   # suma en eje y
        z_centers = 0.5 * (xedges[:-1] + xedges[1:])  # centros de bins en z
        axs2[0].scatter(zplmesh, curr1, label='Current Density' )
        #axs2[0].scatter(z_centers, counts_z / np.max(counts_z) * np.max(np.abs(curr1)), label='Particle Density (scaled)', color='orange', s=10)
        #axs2[0].scatter(z_centers, counts_z, label='Particle Density (scaled)', color='orange', s=10)

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
        
        

        fig4.savefig(lineaf)
        plt.close(fig4)

        #plt.show()


def plot_particles_3plots(protons,run_length,lineaf):
      
      
      #wp.limits(wp.w3d.zmminglobal, wp.w3d.zmmaxglobal, 0., i_beam*2)
        mass_to_ev = (1.67e-27 / 1.6e-19)
        #*13041.02/938.272089

        energy = 0.5 * (protons.getvx()**2 + protons.getvy()**2 + protons.getvz()**2) * mass_to_ev
        print("Ekin max", np.max(energy), "LLLLaverage", np.average(energy))
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
        fig.savefig(lineaf)
        plt.close(fig)


RFQ_TRANSVERSE_Z_FRACTIONS = [0.10, 0.35, 0.65, 0.90]


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


def common_rfq_transverse_z_targets(field_z_min, field_z_max, warp_z_min, warp_z_max, fractions=None):
    """Devuelve posiciones z comunes para comparar cortes XY."""
    if fractions is None:
        fractions = RFQ_TRANSVERSE_Z_FRACTIONS

    z_start = max(float(field_z_min), float(warp_z_min))
    z_end = min(float(field_z_max), float(warp_z_max))
    if z_end <= z_start:
        raise ValueError("No hay rango z comun entre el fieldmap RFQ y la malla de Warp.")

    return np.asarray(
        [
            z_start + fraction * (z_end - z_start)
            for fraction in fractions
        ],
        dtype=float,
    )


def nearest_indices_for_z(z_values, z_targets):
    """Selecciona en una malla los indices mas cercanos a z_targets."""
    z_values = np.asarray(z_values, dtype=float)
    return [
        int(np.argmin(np.abs(z_values - z_target)))
        for z_target in z_targets
    ]


def plot_loaded_rfq_axis_diagnostics(z_values, ez_axis, show=True, output_prefix=None):
    """Grafica Ez y un potencial aproximado sobre el eje del fieldmap cargado."""
    z_values = np.asarray(z_values, dtype=float)
    ez_axis = np.asarray(ez_axis, dtype=float)
    potential_axis = ez_axis * (z_values.max() - z_values.min()) / len(z_values)

    fig_ez, ax_ez = plt.subplots()
    ax_ez.plot(z_values, ez_axis)
    ax_ez.set_xlabel("z (m)")
    ax_ez.set_ylabel("Ez (V/m)")
    ax_ez.set_title("Ez along z-axis at x=y=0")
    ax_ez.grid()

    fig_phi, ax_phi = plt.subplots()
    ax_phi.plot(z_values, potential_axis)
    ax_phi.set_xlabel("z (m)")
    ax_phi.set_ylabel("Potential (V)")

    if output_prefix is not None:
        fig_ez.savefig(f"{output_prefix}_ez_axis.png", dpi=180)
        fig_phi.savefig(f"{output_prefix}_potential_axis.png", dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig_ez)
        plt.close(fig_phi)

    return potential_axis


def plot_rfq_potential_after_first_step_field_loaded(
    ex_rfq,
    ey_rfq,
    ez_rfq,
    x_values,
    y_values,
    z_values,
    dx_rfq,
    dy_rfq,
    dz_rfq,
    voltage_now,
    time_now,
    z_targets,
    output_path,
):
    """Reconstruye y guarda el potencial del campo RFQ aplicado."""
    ix0 = len(x_values) // 2
    iy0 = len(y_values) // 2
    z_indices = nearest_indices_for_z(z_values, z_targets)

    phi_axis = -integrate_from_reference(
        ez_rfq[ix0, iy0, :],
        dz_rfq,
        reference_index=0,
    )

    phi_zx = (
        phi_axis[np.newaxis, :]
        - integrate_from_reference(ex_rfq[:, iy0, :], dx_rfq, ix0, axis=0)
    ) * voltage_now
    phi_zy = (
        phi_axis[np.newaxis, :]
        - integrate_from_reference(ey_rfq[ix0, :, :], dy_rfq, iy0, axis=0)
    ) * voltage_now

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    for axis, iz, z_target in zip(axes.flat[:4], z_indices, z_targets):
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
        axis.set_title(f"Plano XY, z = {z_values[iz]:.3f} m (target {z_target:.3f})")
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
        f"Potencial RFQ despues del paso 1, t={time_now:.4e} s, "
        f"V={voltage_now:.4e} V"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print("RFQ potential diagnostic saved in", output_path)


def plot_rfq_potential_after_first_step_field_warp_internal(
    wp,
    z_targets,
    output_path,
    external_z_values=None,
    external_ez_axis=None,
):
    """Grafica el potencial calculado internamente por Warp."""
    solver = wp.getregisteredsolver()
    x_values = np.linspace(wp.w3d.xmmin, wp.w3d.xmmax, wp.w3d.nx + 1)
    y_values = np.linspace(wp.w3d.ymmin, wp.w3d.ymmax, wp.w3d.ny + 1)
    z_values = np.linspace(wp.w3d.zmmin, wp.w3d.zmmax, wp.w3d.nz + 1)
    z_indices = nearest_indices_for_z(z_values, z_targets)
    ix0 = int(wp.w3d.nx // 2)
    iy0 = int(wp.w3d.ny // 2)

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    for axis, iz, z_target in zip(axes.flat[:4], z_indices, z_targets):
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
        axis.set_title(f"Plano XY, z = {z_values[iz]:.3f} m (target {z_target:.3f})")
        figure.colorbar(image, ax=axis, label="Potencial [V]")

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
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print("Warp internal potential diagnostic saved in", output_path)

    phi_axis = np.asarray(wp.getphi(ix=ix0, iy=iy0, solver=solver))
    electric_field_z = -np.gradient(phi_axis, wp.w3d.dz)
    output_base = str(output_path).rsplit(".", 1)[0]

    fig_phi, ax_phi = plt.subplots()
    ax_phi.scatter(z_values, phi_axis)
    ax_phi.set_xlabel("z (m)")
    ax_phi.set_ylabel("Potential (V)")
    fig_phi.savefig(f"{output_base}_axis_phi.png", dpi=180)
    plt.close(fig_phi)

    fig_ez, ax_ez = plt.subplots()
    ax_ez.plot(z_values, electric_field_z)
    ax_ez.set_xlabel("z (m)")
    ax_ez.set_ylabel("Electric Field Ez (V/m)")
    fig_ez.savefig(f"{output_base}_axis_ez.png", dpi=180)
    plt.close(fig_ez)

    if external_z_values is not None and external_ez_axis is not None:
        norm_internal = np.max(np.abs(electric_field_z))
        norm_external = np.max(np.abs(external_ez_axis))
        if norm_internal > 0.0 and norm_external > 0.0:
            fig_cmp, ax_cmp = plt.subplots()
            ax_cmp.plot(
                external_z_values,
                np.abs(external_ez_axis / norm_external),
                label="external Ez (normalized)",
            )
            ax_cmp.plot(
                z_values,
                np.abs(electric_field_z / norm_internal),
                label="Warp internal Ez (normalized)",
                linestyle="--",
            )
            ax_cmp.set_xlabel("z (m)")
            ax_cmp.set_ylabel("|Ez| normalized")
            ax_cmp.legend()
            fig_cmp.savefig(f"{output_base}_ez_compare.png", dpi=180)
            plt.close(fig_cmp)

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



def _finite2(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]

def _weighted_mean(arr, w):
    if w is None:
        return np.mean(arr)
    return np.sum(w*arr) / np.sum(w)

def _weighted_central_moments(x, xp, w=None):
    """
    Devuelve <x^2>, <x'^2>, <x x'> (momentos centrales, i.e., quitando centroides)
    """
    if w is not None:
        w = np.asarray(w, dtype=float)
        w = w / np.sum(w)
    # centroides
    mx = _weighted_mean(x, w)
    mxp = _weighted_mean(xp, w)
    X = x - mx
    XP = xp - mxp
    if w is None:
        xx  = np.mean(X*X)
        pp  = np.mean(XP*XP)
        xpx = np.mean(X*XP)
    else:
        xx  = np.sum(w*X*X)
        pp  = np.sum(w*XP*XP)
        xpx = np.sum(w*X*XP)
    return xx, pp, xpx, mx, mxp

def twiss_from_arrays(x, xp, w=None, eps_floor=0.0):
    """
    Calcula Twiss (alpha, beta, gamma) y emittancias a partir de x, x'.
    Retorna dict con:
      sigma_x, sigma_xp, cov_x_xp, rho, emit_rms, alpha, beta, gamma, x_mean, xp_mean
    """
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    x, xp = _finite2(x, xp)
    if w is not None:
        w = np.asarray(w, dtype=float)
        w = w[np.isfinite(w)][:len(x)]  # opcional: alinear si ya filtraste

    xx, pp, xpx, mx, mxp = _weighted_central_moments(x, xp, w)
    # emittancia geométrica (RMS)
    disc = xx*pp - xpx**2
    if disc < 0 and np.isclose(disc, 0, atol=1e-20):
        disc = 0.0  # evita negativos por redondeo
    emit = np.sqrt(max(eps_floor, disc))

    # Twiss
    if emit == 0.0:
        # Degenerado: devuelve NaN en Twiss
        alpha = beta = gamma = np.nan
    else:
        beta  = xx / emit
        alpha = -xpx / emit
        gamma = pp / emit

    # extras
    sigma_x  = np.sqrt(xx)
    sigma_xp = np.sqrt(pp)
    rho = xpx / (sigma_x * sigma_xp) if sigma_x>0 and sigma_xp>0 else np.nan

    return {
        "sigma_x": sigma_x,
        "sigma_xp": sigma_xp,
        "cov_x_xp": xpx,
        "rho": rho,
        "emit_rms": emit,      # emittancia geométrica (m·rad)
        "alpha": alpha,
        "beta": beta,          # [m / rad] ≈ [m]
        "gamma": gamma,        # [1/m]
        "x_mean": mx,
        "xp_mean": mxp,
        "n": len(x),
    }

# --- OPCIONAL: emittancia normalizada ---
# Si quieres la normalizada: eps_n = beta_rel * gamma_rel * eps_geom
c  = 299_792_458.0
eV = 1.602176634e-19

def beta_gamma_from_T_classic(T_eV, m_kg):
    """
    Aproximación clásica para beta (v/c) cuando T << mc^2:
      v = sqrt(2T/m), beta = v/c; gamma ~ 1
    Útil para energías realmente no relativistas.
    """
    Tj = np.asarray(T_eV, float) * eV
    v  = np.sqrt(2.0 * Tj / m_kg)
    beta = v / c
    gamma = np.ones_like(beta)
    return beta, gamma

def beta_gamma_from_T_rel(T_eV, m_kg):
    """
    Relativista exacto a partir de T (energía cinética).
    """
    mc2 = m_kg * c**2
    Tj  = np.asarray(T_eV, float) * eV
    Etot = Tj + mc2
    gamma = Etot / mc2
    beta = np.sqrt(1.0 - 1.0/np.maximum(gamma,1.0)**2)
    return beta, gamma

def normalized_emittance(emit_geom, beta_rel, gamma_rel):
    return emit_geom * beta_rel * gamma_rel



#import numpy as np

# Constantes físicas (SI)
MASS_U   = 1.66053906660e-27   # kg
CHARGE_E = 1.602176634e-19     # C

def gaussian_beam_with_emittance(
    N, q_units_e, m_units_u,
    E0_eV,
    a1, b1, e1,
    a2, b2, e2,
    c, dir1, dir2,z1,
    rng=None
):
    """
    Genera un haz 3D gaussiano a partir de Twiss en dos planos ortogonales.
    Devuelve: x, y, z, vx, vy, vz (cada uno ndarray de shape (N,))
    
    Parámetros:
    - N: número de partículas
    - I: corriente total del haz (A) [no se usa en los retornos, sólo FYI]
    - q_units_e: carga en múltiplos de e (p.ej. +1 para protones, -1 para electrones)
    - m_units_u: masa en unidades de u (amu)
    - E0_eV: energía cinética por partícula en eV
    - (a1,b1,e1): Twiss α, β [m/rad], ε (emittance geométrica) para el primer plano
    - (a2,b2,e2): Twiss α, β [m/rad], ε (emittance geométrica) para el segundo plano
    - c: centro (x0,y0,z0) en metros
    - dir1, dir2: vectores directores (3,) que definen el sistema local; dir3 = dir1×dir2
    - rng: np.random.Generator opcional (para reproducibilidad)
    """
    keep_energy=True
    # Masa y carga en SI
    m_si = m_units_u * MASS_U
    q_si = q_units_e * CHARGE_E  # (no se necesita para los retornos de posición/velocidad)

    # Base ortonormal a partir de dir1 y dir2 (como en el C++)
    d1 = np.asarray(dir1, dtype=float)
    d2 = np.asarray(dir2, dtype=float)
    d3 = np.cross(d1, d2)
    d2 = np.cross(d1, d3)

    def _norm(v):
        n = np.linalg.norm(v)
        if n == 0 or not np.isfinite(n):
            raise ValueError("Vectores de dirección inválidos (norma cero o NaN).")
        return v / n

    d1 = _norm(d1)
    d2 = _norm(d2)
    d3 = _norm(d3)

    c = np.asarray(c, dtype=float)
    if c.shape != (3,):
        raise ValueError("c debe ser un vector de 3 componentes (x0,y0,z0).")

    # Velocidad longitudinal a partir de E0 (clásica): v = sqrt(2*E/m)
    vz = np.sqrt(2.0 * E0_eV * CHARGE_E / m_si)
    vz_base = np.sqrt(2.0 * E0_eV * CHARGE_E / m_si)

    # Elipses (rmaj/rmin) y ángulos a partir de Twiss
    def _ellipse_params(a, b, e):
        g = (1.0 + a*a) / b
        h = 0.5*(b + g)
        rmaj = np.sqrt(0.5*e) * (np.sqrt(h+1.0) + np.sqrt(h-1.0))
        rmin = np.sqrt(0.5*e) * (np.sqrt(h+1.0) - np.sqrt(h-1.0))
        theta = 0.5 * np.arctan2(-2.0*a, b - g)
        return rmaj, rmin, theta

    rmaj1, rmin1, theta1 = _ellipse_params(a1, b1, e1)
    rmaj2, rmin2, theta2 = _ellipse_params(a2, b2, e2)

    # RNG
    if rng is None:
        rng = np.random.default_rng()

    # Muestras gaussianas independientes N(0,1)
    rn = rng.normal(size=(N, 4))

    # Escala a radios mayor/menor de cada elipse
    w0 = rmaj1 * rn[:, 0]
    w1 = rmin1 * rn[:, 1]
    w2 = rmaj2 * rn[:, 2]
    w3 = rmin2 * rn[:, 3]

    ct1, st1 = np.cos(theta1), np.sin(theta1)
    ct2, st2 = np.cos(theta2), np.sin(theta2)

    # Coordenadas locales (px[1], px[2], px[3], px[4], px[5]=0, px[6]=vz)
    # p1, v1 (= vz * x1'), p2, v2 (= vz * x2')
    p1 = w0*ct1 - w1*st1
    v1 = vz*(w0*st1 + w1*ct1)

    p2 = w2*ct2 - w3*st2
    v2 = vz*(w2*st2 + w3*ct2)

    p3 = np.zeros(N)          # desplazamiento local a lo largo de d3 (igual que px[5]=0 en C++)
    v3 = np.full(N, vz)       # componente de velocidad a lo largo de d3 (px[6])

    if keep_energy:
        v_perp2 = v1*v1 + v2*v2
        # asegura v3 >= 0 y conserva |v| = vz_base
        np.maximum(vz_base*vz_base - v_perp2, 0.0, out=v3)
        np.sqrt(v3, out=v3)

    # Mapeo a coordenadas del mundo (x, vx, y, vy, z, vz)
    # x = d1*p1 + d2*p2 + d3*p3 + c
    # v = d1*v1 + d2*v2 + d3*v3
    x  = d1[0]*p1 + d2[0]*p2 + d3[0]*p3 + c[0]
    y  = d1[1]*p1 + d2[1]*p2 + d3[1]*p3 + c[1]
    z0  = d1[2]*p1 + d2[2]*p2 + d3[2]*p3 + c[2]
    zm = np.random.uniform(0, z1, N)
    z=z0+zm

    vx = d1[0]*v1 + d2[0]*v2 + d3[0]*v3
    vy = d1[1]*v1 + d2[1]*v2 + d3[1]*v3
    vz_ = d1[2]*v1 + d2[2]*v2 + d3[2]*v3

    return x, y, z, vx, vy, vz_

def time_to_phase_deg(t, f_rf, tref=0.0, phi0_deg=0.0, wrap=True, center180=False):
    """
    t: array-like de tiempos [s]
    f_rf: frecuencia [Hz]
    tref: tiempo de referencia [s] (se usa t - tref)
    phi0_deg: offset de fase [deg]
    wrap: si True, aplica módulo 360
    center180: si True, devuelve en (-180, 180]; requiere wrap=True
    """
    phi = 360.0 * f_rf * (np.asarray(t) - tref) + phi0_deg
    if wrap:
        phi = np.mod(phi, 360.0)
        if center180:
            phi = np.where(phi > 180.0, phi - 360.0, phi)
    return phi
