#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Uso: python dump_pickles.py output.txt

import sys, pickle, numpy as np

fname = sys.argv[1] if len(sys.argv) > 1 else "output.plk"
out_txt = fname + ".dump.txt"

def preview_array(arr, n=10):
    try:
        it = arr.flat
        return [next(it) for _ in range(min(n, arr.size))]
    except Exception:
        return "<no-preview>"

count = 0
with open(fname, "rb") as f, open(out_txt, "w", encoding="utf-8") as w:
    while True:
        try:
            obj = pickle.load(f)
            count += 1
            w.write(f"\n=== OBJETO #{count} ===\n")
            w.write(f"type: {type(obj)}\n")

            # Tuplas tipo ('nombre', valor)
            if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], str):
                k, v = obj
                w.write(f"key: {k!r}\n")
                w.write(f"value_type: {type(v)}\n")
                if isinstance(v, np.ndarray):
                    w.write(f"array shape: {v.shape}  dtype: {v.dtype}\n")
                    w.write(f"preview: {preview_array(v)}\n")
                else:
                    s = repr(v)
                    w.write(f"value: {s[:500]}{'...' if len(s)>500 else ''}\n")

            # Arrays sueltos
            elif isinstance(obj, np.ndarray):
                w.write(f"array shape: {obj.shape}  dtype: {obj.dtype}\n")
                w.write(f"preview: {preview_array(obj)}\n")

            # Diccionarios
            elif isinstance(obj, dict):
                keys = list(obj.keys())
                w.write(f"dict keys ({len(keys)}): {keys[:20]}\n")
                # muestra hasta 3 entradas
                for k in keys[:3]:
                    v = obj[k]
                    if isinstance(v, np.ndarray):
                        w.write(f"  {k}: ndarray shape={v.shape} dtype={v.dtype} preview={preview_array(v)}\n")
                    else:
                        s = repr(v)
                        w.write(f"  {k}: {type(v)} -> {s[:300]}{'...' if len(s)>300 else ''}\n")

            # Listas/tuplas genéricas
            elif isinstance(obj, (list, tuple)):
                w.write(f"{type(obj)} len={len(obj)}\n")
                for i, v in enumerate(obj[:5]):
                    s = repr(v)
                    w.write(f"  [{i}] {type(v)} -> {s[:300]}{'...' if len(s)>300 else ''}\n")

            else:
                s = repr(obj)
                w.write(f"value: {s[:500]}{'...' if len(s)>500 else ''}\n")

        except EOFError:
            break
        except Exception as e:
            w.write(f"\n--- Fin de pickles (bloque binario no-pickle o error): {e} ---\n")
            break

print(f"Hecho. Resumen legible en: {out_txt} (objetos leídos: {count})")
