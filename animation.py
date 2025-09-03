
# Obtener lista de archivos ordenados
#imagenes = sorted(glob.glob("step*.png"))
imagenes = [f"step_{i}.png" for i in range(0, 945, 5)]
# Leer la primera imagen para obtener tamaño


from moviepy.editor import ImageSequenceClip

# Supón que tienes esto:
# nombres_existentes = ['step10.png', 'step20.png', 'step30.png', ..., 'step200.png']

# Crea el video clip
clip = ImageSequenceClip(imagenes, fps=5)  # Cambia fps según lo que quieras

# Guarda el video en formato mp4
clip.write_videofile("video.mp4")


print("Video creado: output.mp4")


