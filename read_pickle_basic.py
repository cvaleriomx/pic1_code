import pandas as pd
import numpy as np


import pickle, pprint, json

with open("output.plk", "rb") as f:
    data = pickle.load(f)

with open("output_dump.txt", "w") as fout:
    pprint.pprint(data, stream=fout, width=120)

with open("output_dump.json", "w") as fout:
    json.dump(data, fout, indent=2, default=str)

filename="output.plk"
def inspect_pickle(filename, preview=5):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    print("="*50)
    print(f"Archivo: {filename}")
    print(f"Tipo: {type(data)}")
    print("="*50)

    # Caso: DataFrame
    if isinstance(data, pd.DataFrame):
        print("Contenido: DataFrame")
        print(data.info())
        print("\nPrimeras filas:")
        print(data.head(preview))

    # Caso: Diccionario
    elif isinstance(data, dict):
        print(f"Contenido: Diccionario con {len(data)} claves")
        print("Claves:", list(data.keys())[:10])  # muestra primeras 10
        for k in list(data.keys())[:3]:  # preview de hasta 3 claves
            print(f"\n[{k}] -> tipo {type(data[k])}")
            try:
                print(repr(data[k])[:200])  # preview corto
            except:
                pass

    # Caso: Lista o tupla
    elif isinstance(data, (list, tuple)):
        print(f"Contenido: {type(data)} con {len(data)} elementos")
        print("Primeros elementos:")
        for i, elem in enumerate(data[:preview]):
            print(f"[{i}] -> {type(elem)} : {repr(elem)[:200]}")

    # Caso: numpy array
    elif isinstance(data, np.ndarray):
        print("Contenido: Numpy array")
        print("Forma:", data.shape)
        print("Dtype:", data.dtype)
        print("Primeros valores:", data.flat[:preview])

    # Otro caso
    else:
        print("Contenido no estándar:")
        try:
            print(repr(data)[:500])
        except:
            print("No se puede mostrar directamente.")

    return data  # lo regresa por si lo quieres manipular después

# Uso:
obj = inspect_pickle("output.plk")
#data = ('lsavephi', True)  # ejemplo de lo que te salió







def load_and_show(filename, preview=10):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    print("="*60)
    print(f"Archivo: {filename}")
    print("Tipo:", type(data))
    print("="*60)

    # Si es diccionario
    if isinstance(data, dict):
        print("Diccionario con claves:", list(data.keys()))
        for k, v in list(data.items())[:3]:  # muestra hasta 3 claves
            print(f"\nClave: {k} -> tipo {type(v)}")
            if isinstance(v, np.ndarray):
                print("  Forma:", v.shape, "Dtype:", v.dtype)
                print("  Preview:", v.flat[:preview])
            else:
                print("  Valor:", repr(v)[:200])

    # Si es array numpy
    elif isinstance(data, np.ndarray):
        print("Array numpy")
        print("Forma:", data.shape, "Dtype:", data.dtype)
        print("Primeros valores:", data.flat[:preview])

    # Si es lista o tupla
    elif isinstance(data, (list, tuple)):
        print(f"{type(data)} con {len(data)} elementos")
        for i, v in enumerate(data[:preview]):
            print(f"[{i}] tipo {type(v)} -> {repr(v)[:200]}")

    # Si es algo raro
    else:
        print("Contenido:")
        print(repr(data)[:500])

    return data

# === USO ===
data = load_and_show("output.plk")
# data = load_and_show("output001000.pdump")

import pickle, numpy as np

fname = "output.plk"   # tu archivo grande (~2 GB)

objs = []
with open(fname, "rb") as f:
    while True:
        try:
            o = pickle.load(f)   # lee el siguiente objeto pickle (si hay varios)
            objs.append(o)
            print(f"Leído objeto #{len(objs)} -> {type(o)}")
        except EOFError:
            break

print("\nTotal objetos leídos:", len(objs))

# Explora cada objeto y muestra datos reales si hay arrays o dicts
for i, o in enumerate(objs):
    print("\n", "="*30, f"Objeto #{i+1}", "="*30)
    if isinstance(o, np.ndarray):
        print("ndarray -> shape:", o.shape, "dtype:", o.dtype)
        # preview seguro sin cargar todo:
        it = o.flat
        preview = [next(it) for _ in range(min(10, o.size))]
        print("Preview:", preview)
    elif isinstance(o, dict):
        print("dict -> claves:", list(o.keys())[:20])
        # muestra hasta 3 entradas con resumen
        for k in list(o.keys())[:3]:
            v = o[k]
            print(f"  {k}: {type(v)}", end="")
            if isinstance(v, np.ndarray):
                print("  shape:", v.shape, "dtype:", v.dtype)
                it = v.flat
                pv = [next(it) for _ in range(min(10, v.size))]
                print("     preview:", pv)
            else:
                print("  valor:", repr(v)[:200])
    elif isinstance(o, (list, tuple)):
        print(f"{type(o)} de longitud {len(o)}")
        for j, v in enumerate(o[:5]):  # preview
            print(f"  [{j}] {type(v)} -> {repr(v)[:200]}")
    else:
        print(type(o), repr(o)[:500])
import numpy as np
from collections import Counter

print("\n=== Resumen de los primeros 5 objetos ===")
for i, o in enumerate(objs[:5]):
    print(f"\nObjeto #{i+1}: {type(o)}")
    if isinstance(o, tuple):
        print("  len:", len(o))
        if len(o) == 2 and isinstance(o[0], str):
            k, v = o
            print(f"  clave: {k!r}  tipo(valor): {type(v)}")
            if isinstance(v, np.ndarray):
                print("  array -> shape:", v.shape, "dtype:", v.dtype, "preview:", v.flat[:10])
            else:
                print("  valor:", repr(v)[:200])
        else:
            # tupla genérica
            for j, sub in enumerate(o[:5]):
                print(f"  [{j}] {type(sub)} -> {repr(sub)[:200]}")
    else:
        print("  contenido:", repr(o)[:200])

# Conteo de claves tipo ('nombre', valor)
keys = [o[0] for o in objs if isinstance(o, tuple) and len(o)==2 and isinstance(o[0], str)]
print("\nTotal tuplas tipo ('nombre', valor):", len(keys))
print("Top claves:", Counter(keys).most_common(10))

