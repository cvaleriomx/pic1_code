# --- Guardar cruces de un subconjunto definido por filtros espaciales ---
#   - Filtra por caja en x,y (xmin<=x<=xmax, ymin<=y<=ymax)
#   - Filtra por lado en z respecto al plano (z<z0 o z>z0), o por rango zmin<=z<=zmax
#   - Rastrea por PID para no recorrer todas las partículas
#
# Uso típico:
#   z0 = 0.1
#   monitor = PlaneCrossSaverFiltered(
#       species=protons, z0=z0, filename="crossings_z0p10.csv",
#       xlim=(-5e-3, 5e-3), ylim=(-5e-3, 5e-3),
#       z_side="below",   # "below" = solo las que están con z < z0 al inicio
#       reseed_each_step=False  # True si hay inyección y quieres captar nuevas que entren al filtro
#   )
#   installafterstep(monitor)

import numpy as np
from warp import top

class PlaneCrossSaverFiltered(object):
    def __init__(self, species, z0, filename,
                 xlim=None, ylim=None,
                 z_side=None,      # "below", "above" o None
                 zrange=None,      # tuple (zmin, zmax) alternativa a z_side
                 reseed_each_step=False,  # volver a adicionar PIDs que entren al filtro en cada paso
                 include_dir=True,       # guardar dirección del cruce
                 require_pid=True,       # forzar uso de PID
                 save_header=True):
        """
        xlim, ylim: (min, max) o None para no filtrar por esa coordenada.
        z_side: "below" (solo z<z0), "above" (solo z>z0) o None.
        zrange: (zmin, zmax) prioriza sobre z_side si se provee.
        reseed_each_step: si True, agrega PIDs nuevos que cumplan el filtro en cada paso.
        """
        self.sp = species
        self.z0 = float(z0)
        self.filename = filename
        self.include_dir = include_dir
        self.reseed_each_step = reseed_each_step
        self.require_pid = require_pid

        self.xlim = xlim
        self.ylim = ylim
        self.z_side = z_side
        self.zrange = zrange

        # Estado
        self._has_prev = False
        self.prev = {}             # buffers de estado previo SOLO para activos
        self.active_pids = set()   # PIDs activos (aún no cruzaron)
        self.initialized = False

        # Encabezado CSV
        if save_header:
            cols = ["t_cross","x","y","z","vx","vy","vz","pid"]
            if include_dir: cols.append("dir")
            with open(self.filename, "w") as f:
                f.write(",".join(cols) + "\n")

    # -------- utilidades internas ----------

    def _get_arrays_now(self):
        """Trae arrays actuales de la especie."""
        x  = np.asarray(self.sp.getx())
        y  = np.asarray(self.sp.gety())
        z  = np.asarray(self.sp.getz())
        vx = np.asarray(self.sp.getvx())
        vy = np.asarray(self.sp.getvy())
        vz = np.asarray(self.sp.getvz())
        print("N partículas ahora:", len(x))
        if not hasattr(self.sp, "getpid"):
            if self.require_pid:
                raise RuntimeError("Esta clase requiere PIDs: habilita npid>0 o usa la versión sin PID (fallback más abajo).")
            pid = None
        else:
            pid = np.asarray(self.sp.getpid())
        return x,y,z,vx,vy,vz,pid

    def _filter_mask(self, x, y, z):
        """Máscara booleana de las que cumplen el filtro espacial."""
        m = np.ones_like(z, dtype=bool)
        if self.xlim is not None:
            m &= (x >= self.xlim[0]) & (x <= self.xlim[1])
        if self.ylim is not None:
            m &= (y >= self.ylim[0]) & (y <= self.ylim[1])

        if self.zrange is not None:
            zmin, zmax = self.zrange
            m &= (z >= zmin) & (z <= zmax)
        elif self.z_side == "below":
            m &= (z < self.z0)
        elif self.z_side == "above":
            m &= (z > self.z0)
        # si z_side es None, no restringe z

        return m

    def _pid_index_map(self, pid):
        """Devuelve mapping pid->index usando argsort (O(n log n), pero vectorizado)."""
        order = np.argsort(pid)
        return pid[order], order

    def _select_by_pids(self, pid, target_set):
        """Devuelve índices de 'pid' que están en target_set (rápido con búsqueda binaria)."""
        # ordenamos pids y hacemos búsqueda con np.searchsorted
        sorted_pids, order = self._pid_index_map(pid)
        target = np.fromiter(target_set, dtype=pid.dtype, count=len(target_set))
        pos = np.searchsorted(sorted_pids, target)
        # pos dentro de rango y coincide el valor => índice válido
        in_range = (pos >= 0) & (pos < sorted_pids.size)
        pos = pos[in_range]
        match = sorted_pids[pos] == target[in_range]
        pos = pos[match]
        # convertir de índice en array ordenado a índice original
        return order[pos]

    # ------------- ciclo por paso ---------------

    def __call__(self):
        x,y,z,vx,vy,vz,pid = self._get_arrays_now()
        n = z.size
        if n == 0:
            self._has_prev = False
            return

        # Inicializar conjunto activo
        if not self.initialized or (self.reseed_each_step and not self._has_prev):
            mask0 = self._filter_mask(x,y,z)
            if pid is None:
                raise RuntimeError("Se requieren PIDs para esta estrategia eficiente. (Ver fallback sin PID más abajo).")
            self.active_pids |= set(pid[mask0].tolist())
            self.initialized = True

        # Reseed en cada paso (para inyección continua)
        if self.reseed_each_step:
            mask_add = self._filter_mask(x,y,z)
            self.active_pids |= set(pid[mask_add].tolist())

        # Si aún no hay estado previo, cachea SOLO activos y sale
        if not self._has_prev:
            if len(self.active_pids) == 0:
                return
            idx_act = self._select_by_pids(pid, self.active_pids)
            self.prev = dict(
                pid=pid[idx_act].copy(),
                x=x[idx_act].copy(), y=y[idx_act].copy(), z=z[idx_act].copy(),
                vx=vx[idx_act].copy(), vy=vy[idx_act].copy(), vz=vz[idx_act].copy()
            )
            self._has_prev = True
            return

        # Construir vectores "previos" y "actuales" alineados por PID activo
        if len(self.active_pids) == 0:
            return

        # índices actuales de PIDs activos
        idx_now = self._select_by_pids(pid, self.active_pids)
        if idx_now.size == 0:
            return

        # mapping de prev.pid -> posición actual vía búsqueda por PID
        pid_now = pid[idx_now]
        order_now = np.argsort(pid_now)
        pid_now_sorted = pid_now[order_now]

        # para cada pid previo, encontrar su índice actual (si sigue existiendo)
        pid_prev = self.prev["pid"]
        pos = np.searchsorted(pid_now_sorted, pid_prev)
        valid = (pos >= 0) & (pos < pid_now_sorted.size)
        pos = pos[valid]
        match = pid_now_sorted[pos] == pid_prev[valid]
        pos = pos[match]

        # si no queda ninguno, rehacer estado y salir
        if pos.size == 0:
            # reconstruir prev con actuales activos para el siguiente paso
            self.prev = dict(
                pid=pid_now.copy(),
                x=x[idx_now].copy(), y=y[idx_now].copy(), z=z[idx_now].copy(),
                vx=vx[idx_now].copy(), vy=vy[idx_now].copy(), vz=vz[idx_now].copy()
            )
            return

        # obtener pares prev/now alineados
        sel_prev = np.where(valid)[0][match]          # índices en prev.*
        sel_now  = order_now[pos]                     # índices en idx_now -> ahora

        x_prev  = self.prev["x"][sel_prev]
        y_prev  = self.prev["y"][sel_prev]
        z_prev  = self.prev["z"][sel_prev]
        vx_prev = self.prev["vx"][sel_prev]
        vy_prev = self.prev["vy"][sel_prev]
        vz_prev = self.prev["vz"][sel_prev]
        pid_sel = self.prev["pid"][sel_prev]

        x_now  = x[idx_now][sel_now]
        y_now  = y[idx_now][sel_now]
        z_now  = z[idx_now][sel_now]
        vx_now = vx[idx_now][sel_now]
        vy_now = vy[idx_now][sel_now]
        vz_now = vz[idx_now][sel_now]

        # Detección de cruce (cambio de signo respecto a z0)
        dz_prev = z_prev - self.z0
        dz_now  = z_now  - self.z0
        moved = (z_now != z_prev) & np.isfinite(dz_prev) & np.isfinite(dz_now)
        crossed = (dz_prev * dz_now <= 0.0) & moved

        if np.any(crossed):
            I = np.where(crossed)[0]
            alpha = (self.z0 - z_prev[I]) / (z_now[I] - z_prev[I])
            alpha = np.clip(alpha, 0.0, 1.0)

            x_cross  = x_prev[I]  + alpha * (x_now[I]  - x_prev[I])
            y_cross  = y_prev[I]  + alpha * (y_now[I]  - y_prev[I])
            z_cross  = np.full_like(alpha, self.z0, dtype=float)
            vx_cross = vx_prev[I] + alpha * (vx_now[I] - vx_prev[I])
            vy_cross = vy_prev[I] + alpha * (vy_now[I] - vy_prev[I])
            vz_cross = vz_prev[I] + alpha * (vz_now[I] - vz_prev[I])

            # tiempo interpolado
            dt = getattr(top, "dt", None)
            t_now = getattr(top, "time", None)
            if (dt is not None) and (t_now is not None):
                t_cross = (t_now - dt) + alpha * dt
            else:
                t_cross = np.full_like(alpha, np.nan, dtype=float)

            # dirección
            if self.include_dir:
                dir_sign = np.sign(vz_cross)
                dir_sign[dir_sign == 0] = 1
            else:
                dir_sign = None

            # escribir filas
            with open(self.filename, "a") as f:
                for k in range(I.size):
                    row = [t_cross[k], x_cross[k], y_cross[k], z_cross[k],
                           vx_cross[k], vy_cross[k], vz_cross[k], int(pid_sel[I[k]])]
                    if dir_sign is not None:
                        row.append(int(dir_sign[k]))
                    f.write(",".join(str(v) for v in row) + "\n")

            # eliminar de activos los que acaban de cruzar
            crossed_pids = set(pid_sel[I].tolist())
            self.active_pids -= crossed_pids

        # reconstruir prev SOLO con PIDs activos restantes (y quizá nuevos si reseed)
        # 1) PIDs activos actuales (después de remover cruzados)
        if len(self.active_pids):
            idx_now2 = self._select_by_pids(pid, self.active_pids)
            self.prev = dict(
                pid=pid[idx_now2].copy(),
                x=x[idx_now2].copy(), y=y[idx_now2].copy(), z=z[idx_now2].copy(),
                vx=vx[idx_now2].copy(), vy=vy[idx_now2].copy(), vz=vz[idx_now2].copy()
            )
            self._has_prev = True
        else:
            self._has_prev = False  # no hay nada que rastrear

# ---------------- Fallback SIN PID (menos robusto) ----------------
# Si no tienes PIDs, puedes volver al monitor simple y filtrar cada paso:
#  - Toma solo índices que cumplen el filtro en 'prev' y en 'now'.
#  - ¡OJO! Si hay reordenamientos o pérdida/inyección, puede desalinear pares.
#
# Te lo paso si lo necesitas; en general, habilitar PIDs es muy recomendable.
