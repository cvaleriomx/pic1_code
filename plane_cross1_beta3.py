# -*- coding: utf-8 -*-
import numpy as np
from warp import top

class PlaneCrossSaverFiltered:
    def __init__(self, species, z0, filename,
                 xlim=None, ylim=None,
                 z_side=None,         # "below", "above" o None
                 zrange=None,         # (zmin, zmax) si prefieres ventana finita
                 reseed_each_step=False,
                 include_dir=True,
                 require_pid=True,
                 save_header=True,
                 start_step=0, end_step=None,
                 debug=False, dz_probe=None):
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
        self.debug = debug
        self.dz_probe = dz_probe
        self._has_prev = False
        self.prev = {}
        self.active_pids = set()
        self.initialized = False
        self._step_count = 0
        self.start_step = start_step
        self.end_step = end_step
        

        if save_header:
            cols = ["t_cross","x","y","z","vx","vy","vz","pid"]
            if include_dir: cols.append("dir")
            with open(self.filename, "w") as f:
                f.write(",".join(cols) + "\n")
     


    # ---------- helpers ----------
    def _get_arrays_now(self):
        x  = np.asarray(self.sp.getx())
        y  = np.asarray(self.sp.gety())
        z  = np.asarray(self.sp.getz())
        vx = np.asarray(self.sp.getvx())
        vy = np.asarray(self.sp.getvy())
        vz = np.asarray(self.sp.getvz())
        if not hasattr(self.sp, "getpid"):
            if self.require_pid:
                raise RuntimeError("Se requieren PIDs (npid>0) o usa require_pid=False.")
            pid = None
        else:
            pid = np.asarray(self.sp.getpid())
        return x,y,z,vx,vy,vz,pid

    def _filter_mask(self, x, y, z):
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
        return m

    def _pid_index_map(self, pid):
        order = np.argsort(pid)
        return pid[order], order

    def _select_by_pids(self, pid, target_set):
        if len(target_set) == 0:
            return np.array([], dtype=int)
        sorted_pids, order = self._pid_index_map(pid)
        target = np.fromiter(target_set, dtype=pid.dtype, count=len(target_set))
        pos = np.searchsorted(sorted_pids, target)
        in_range = (pos >= 0) & (pos < sorted_pids.size)
        pos = pos[in_range]
        match = sorted_pids[pos] == target[in_range]
        pos = pos[match]
        return order[pos]

    # ---------- hook para afterstep ----------
    def step_monitor(self):
        self._step_count += 1
        # --- filtro por rango de steps ---
        if self._step_count < self.start_step:
            return
        if (self.end_step is not None) and (self._step_count > self.end_step):
            return



        x,y,z,vx,vy,vz,pid = self._get_arrays_now()
        n = z.size
        print(f"[PlaneCross] step {self._step_count}: Npart={n}", flush=True)
        if n == 0:
            self._has_prev = False
            if self.debug: print("[PlaneCross] vacío en paso", self._step_count, flush=True)
            return

        # inicialización / reseed
        if not self.initialized or (self.reseed_each_step and not self._has_prev):
            mask0 = self._filter_mask(x,y,z)
            if pid is None:
                raise RuntimeError("Sin PID con require_pid=True.")
            self.active_pids |= set(pid[mask0].tolist())
            self.initialized = True

        if self.reseed_each_step:
            mask_add = self._filter_mask(x,y,z)
            self.active_pids |= set(pid[mask_add].tolist())

        if self.debug:
            m0 = self._filter_mask(x,y,z)
            zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
            print(f"[PlaneCross] step {self._step_count}: N={n}, filtro={int(m0.sum())}, activos={len(self.active_pids)}, z0={self.z0}, z∈[{zmin:.3g},{zmax:.3g}]", flush=True)

        # primera vez: cachea prev y sal
        if not self._has_prev:
            if len(self.active_pids) == 0: return
            idx_act = self._select_by_pids(pid, self.active_pids)
            if idx_act.size == 0: return
            self.prev = dict(
                pid=pid[idx_act].copy(),
                x=x[idx_act].copy(), y=y[idx_act].copy(), z=z[idx_act].copy(),
                vx=vx[idx_act].copy(), vy=vy[idx_act].copy(), vz=vz[idx_act].copy()
            )
            self._has_prev = True
            return

        if len(self.active_pids) == 0:
            self._has_prev = False
            return

        idx_now = self._select_by_pids(pid, self.active_pids)
        if idx_now.size == 0:
            self._has_prev = False
            return

        pid_now = pid[idx_now]
        order_now = np.argsort(pid_now)
        pid_now_sorted = pid_now[order_now]

        pid_prev = self.prev["pid"]
        pos = np.searchsorted(pid_now_sorted, pid_prev)
        valid = (pos >= 0) & (pos < pid_now_sorted.size)
        pos = pos[valid]
        match = pid_now_sorted[pos] == pid_prev[valid]
        pos = pos[match]
        if pos.size == 0:
            # rehace prev con actuales
            self.prev = dict(
                pid=pid_now.copy(),
                x=x[idx_now].copy(), y=y[idx_now].copy(), z=z[idx_now].copy(),
                vx=vx[idx_now].copy(), vy=vy[idx_now].copy(), vz=vz[idx_now].copy()
            )
            return

        sel_prev = np.where(valid)[0][match]
        sel_now  = order_now[pos]

        x_prev,y_prev,z_prev = self.prev["x"][sel_prev], self.prev["y"][sel_prev], self.prev["z"][sel_prev]
        vx_prev,vy_prev,vz_prev = self.prev["vx"][sel_prev], self.prev["vy"][sel_prev], self.prev["vz"][sel_prev]
        pid_sel = self.prev["pid"][sel_prev]

        x_now,y_now,z_now = x[idx_now][sel_now], y[idx_now][sel_now], z[idx_now][sel_now]
        vx_now,vy_now,vz_now = vx[idx_now][sel_now], vy[idx_now][sel_now], vz[idx_now][sel_now]


# --- detección de cruces ---
        dz_prev = z_prev - self.z0
        dz_now  = z_now  - self.z0
        moved   = (z_now != z_prev) & np.isfinite(dz_prev) & np.isfinite(dz_now)

        # con tolerancia opcional
        eps = 0.0  # o 1e-9 si quieres holgura numérica
        up_cross   = (z_prev < self.z0 - eps) & (z_now >= self.z0 + eps) & moved
        down_cross = (z_prev > self.z0 + eps) & (z_now <= self.z0 - eps) & moved
        crossed    = up_cross  # o (up_cross | down_cross) si quieres ambos sentidos

        if self.debug:
            print(f"[PlaneCross] pares={sel_prev.size}, cruzan={int(crossed.sum())}", flush=True)

        # === TODO LO QUE SIGUE SOLO SI HAY CRUCES ===
        if np.any(crossed):
            I = np.where(crossed)[0]

            # interpolación lineal al plano
            alpha = (self.z0 - z_prev[I]) / (z_now[I] - z_prev[I])
            alpha = np.clip(alpha, 0.0, 1.0)

            x_cross  = x_prev[I]  + alpha * (x_now[I]  - x_prev[I])
            y_cross  = y_prev[I]  + alpha * (y_now[I]  - y_prev[I])
            z_cross  = np.full_like(alpha, self.z0, dtype=float)
            vx_cross = vx_prev[I] + alpha * (vx_now[I] - vx_prev[I])
            vy_cross = vy_prev[I] + alpha * (vy_now[I] - vy_prev[I])
            vz_cross = vz_prev[I] + alpha * (vz_now[I] - vz_prev[I])
            pid_cross = pid_sel[I]

            # tiempo interpolado
            dt   = getattr(top, "dt", None)
            tnow = getattr(top, "time", None)
            if (dt is not None) and (tnow is not None):
                t_cross = (tnow - dt) + alpha * dt
            else:
                t_cross = np.full_like(alpha, np.nan, dtype=float)

            # dirección (si se pidió)
            dir_sign = None
            if self.include_dir:
                dir_sign = np.sign(vz_cross)
                dir_sign[dir_sign == 0] = 1

            # --- FILTRO x,y EN EL INSTANTE DE CRUCE (garantiza respetar ROI) ---
            if (self.xlim is not None) or (self.ylim is not None):
                keep = np.ones_like(alpha, dtype=bool)
                if self.xlim is not None:
                    keep &= (x_cross >= self.xlim[0]) & (x_cross <= self.xlim[1])
                if self.ylim is not None:
                    keep &= (y_cross >= self.ylim[0]) & (y_cross <= self.ylim[1])

                if not np.all(keep):
                    x_cross  = x_cross[keep]
                    y_cross  = y_cross[keep]
                    z_cross  = z_cross[keep]
                    vx_cross = vx_cross[keep]
                    vy_cross = vy_cross[keep]
                    vz_cross = vz_cross[keep]
                    t_cross  = t_cross[keep]
                    pid_cross = pid_cross[keep]
                    if self.include_dir and (dir_sign is not None):
                        dir_sign = dir_sign[keep]

            # si quedó algo tras el filtro, escribir CSV
            if x_cross.size > 0:
                with open(self.filename, "a") as f:
                    for k in range(x_cross.size):
                        row = [t_cross[k], x_cross[k], y_cross[k], z_cross[k],
                            vx_cross[k], vy_cross[k], vz_cross[k], int(pid_cross[k])]
                        if self.include_dir and (dir_sign is not None):
                            row.append(int(dir_sign[k]))
                        f.write(",".join(str(v) for v in row) + "\n")

                # eliminar de activos SOLO los PIDs que realmente escribimos (pid_cross)
                self.active_pids -= set(pid_cross.astype(int).tolist())

        

           

            # quita de activos los que ya cruzaron
            #crossed_pids = set(pid_sel[I].tolist())
            #self.active_pids -= crossed_pids

        # sonda cerca del plano (diagnóstico)
        if (not np.any(crossed)) and (self.dz_probe is not None) and self.debug:
            near = np.where(np.abs(dz_now) <= self.dz_probe)[0]
            if near.size:
                print(f"[PlaneCross][SONDA] {near.size} con |z−z0| <= {self.dz_probe}", flush=True)

        # reconstruye prev con activos restantes
        if len(self.active_pids):
            idx_now2 = self._select_by_pids(pid, self.active_pids)
            if idx_now2.size:
                self.prev = dict(
                    pid=pid[idx_now2].copy(),
                    x=x[idx_now2].copy(), y=y[idx_now2].copy(), z=z[idx_now2].copy(),
                    vx=vx[idx_now2].copy(), vy=vy[idx_now2].copy(), vz=vz[idx_now2].copy()
                )
                self._has_prev = True
            else:
                self._has_prev = False
        else:
            self._has_prev = False
