from collections.abc import Callable
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import my_functions as mf

type ImagenArray = np.ndarray[np.dtype[np.float64]]


def filtro_raiz(y: ImagenArray) -> ImagenArray:
    return np.sqrt(np.clip(y, 0.0, 1.0))


def filtro_cuadrado(y: ImagenArray) -> ImagenArray:
    return np.square(np.clip(y, 0.0, 1.0))


def filtro_lineal_trozos(y: ImagenArray, y_min: float, y_max: float) -> ImagenArray:
#    -Y < y_min            -->  Y' = 0
#    -y_min <= Y <= y_max  --> Y' = (Y - y_min) / (y_max - y_min)
#    -Y > y_max            -->   Y' = 1 
    return np.clip((y - y_min) / (y_max - y_min), 0.0, 1.0)   


def filtro_lineal_trozos2(y: ImagenArray, y_min: float = 0.2, y_max: float = 1.0) -> ImagenArray:
# Filtro lineal a trozos.
# cada par (x, y) indica: luminancia original -> luminancia resultante.
# para cambiar el contraste hay que modificar lost puntos.
    puntos_x = [y_min, y_max]
    puntos_y = [0.0, 1.0]    
# left=0.0 cubre Y < y_min y right=1.0 cubre Y > y_max
    return np.interp(y, puntos_x, puntos_y, left=0.0, right=1.0)


FILTROS: dict[str, Callable[[ImagenArray], ImagenArray]] = {
    "Raíz": filtro_raiz,
    "Cuadrado": filtro_cuadrado,
    "Lineal a trozos": filtro_lineal_trozos,
}


class AppFiltrosYIQ:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("PDI TP3 - Filtros sobre luminancia Y")
        self.ventana.geometry("1200x650")

        self.imagen_original = None
        self.imagen_procesada = None
        self.imagen_yiq_mod = None

        self.crear_interfaz()

    def crear_interfaz(self):
        panel = ttk.Frame(self.ventana)
        panel.pack(fill="both", expand=True, padx=10, pady=10)

        frame_imagenes = ttk.Frame(panel)
        frame_imagenes.pack_propagate(False)
        frame_imagenes.pack(side="left", fill="both", expand=True, padx=10)

        self.label_original = ttk.Label(
            frame_imagenes,
            text="Imagen original",
            anchor="center",
            background="#dddddd",
        )
        self.label_original.pack(side="left", fill="both", expand=True, padx=5)

        self.label_procesada = ttk.Label(
            frame_imagenes,
            text="Imagen procesada",
            anchor="center",
            background="#dddddd",
        )
        self.label_procesada.pack(side="left", fill="both", expand=True, padx=5)

        controles = ttk.Frame(panel)
        controles.pack(side="right", fill="y", padx=10)

        ttk.Label(controles, text="Filtro sobre Y", font=("Arial", 14, "bold")).pack(
            pady=10
        )

        self.filtro_var = tk.StringVar(value="Raíz")


        for opcion in FILTROS.keys():
            ttk.Radiobutton(
                controles, text=opcion, variable=self.filtro_var, value=opcion
            ).pack(anchor="w")
        
        ttk.Label(controles, text="Ingrese Y_min").pack()  
        self.y_min_var = tk.DoubleVar(value=0.2)
        ttk.Entry(controles ,textvariable=self.y_min_var).pack(fill="x", pady=5)
        
        ttk.Label(controles, text="Ingrese Y_max").pack()  
        self.y_max_var = tk.DoubleVar(value=0.8)
        ttk.Entry(controles ,textvariable=self.y_max_var).pack(fill="x", pady=5)
        
        ttk.Button(controles, text="Abrir imagen", command=self.abrir_imagen).pack(
            fill="x", pady=5
        )
        ttk.Button(controles, text="Aplicar filtro", command=self.aplicar_filtro).pack(
            fill="x", pady=5
        )
        ttk.Button(controles, text="Guardar como", command=self.guardar_imagen).pack(
            fill="x", pady=5
        )

        ttk.Button(controles, text="Restaurar", command=self.restaurar).pack(
            fill="x", pady=5
        )

        ttk.Button(controles, text="Histograma", command=self.mostrar_histograma).pack(
                            fill="x", pady=5
                )
        self.estado = ttk.Label(self.ventana, text="Listo.", anchor="w")
        self.estado.pack(side="bottom", fill="x", padx=10, pady=10)

    def abrir_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if not ruta:
            return

        imagen_pil = Image.open(ruta).convert("RGB")
        self.imagen_original = np.array(imagen_pil) / 255.0
        self.imagen_procesada = self.imagen_original.copy()
        self.imagen_yiq_mod = None

        self.mostrar_imagen(self.imagen_original, self.label_original)
        self.mostrar_imagen(self.imagen_procesada, self.label_procesada)
        self.estado.config(text="Imagen cargada y normalizada a [0,1].")

    def aplicar_filtro(self):
        if self.imagen_original is None:
            messagebox.showwarning("Atención", "Primero abrí una imagen.")
            return

        yiq = mf.rgb_a_yiq(self.imagen_original)

        # copiar para modificar solo Y y conservar IQ.
        yiq_mod = yiq.copy()
        filtro = self.filtro_var.get()

        # Calcular Y' = f(Y) según el filtro elegido
        
        if filtro == "Lineal a trozos":
            y_min = self.y_min_var.get()
            y_max = self.y_max_var.get()
                        
            if y_min >= y_max:
                messagebox.showwarning("Atención", "Y_min debe ser menor que Y_max.")
                return
                        
            yiq_mod[:, :, 0] = filtro_lineal_trozos(yiq_mod[:, :, 0], y_min, y_max)
        
        elif (fn_filtro := FILTROS.get(filtro)) is not None:
            yiq_mod[:, :, 0] = fn_filtro(yiq_mod[:, :, 0])

        # Convertir Y'IQ de nuevo a RGB
        self.imagen_yiq_mod = yiq_mod
        self.imagen_procesada = mf.yiq_a_rgb(yiq_mod)

        self.mostrar_imagen(self.imagen_procesada, self.label_procesada)
        self.estado.config(text=f"Filtro aplicado: {filtro}")

    def mostrar_imagen(self, array_imagen, label_widget):
        imagen_uint8 = (np.clip(array_imagen, 0, 1) * 255).astype(np.uint8)
        imagen_pil = Image.fromarray(imagen_uint8)
        imagen_pil.thumbnail((500, 500))

        foto = ImageTk.PhotoImage(imagen_pil)
        label_widget.foto = foto  
        label_widget.config(image=foto, text="")

    def guardar_imagen(self):
        if self.imagen_procesada is None:
            messagebox.showwarning("Atención", "No hay imagen procesada para guardar.")
            return

        ruta = filedialog.asksaveasfilename(
            initialfile="imagen_filtro_yiq.png",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("TIFF", "*.tiff"),
                ("BMP", "*.bmp"),
            ],
        )
        if not ruta:
            return

        imagen_uint8 = (np.clip(self.imagen_procesada, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(imagen_uint8).save(ruta)
        self.estado.config(text="Imagen guardada.")

    def restaurar(self):
        if self.imagen_original is None:
            return

        self.imagen_procesada = self.imagen_original.copy()
        self.imagen_yiq_mod = None
        self.mostrar_imagen(self.imagen_procesada, self.label_procesada)
        self.estado.config(text="Imagen original restaurada.")
    
    def mostrar_histograma(self):
            # Calcular canal Y de la imagen original
            yiq_original = mf.rgb_a_yiq(self.imagen_original)
            y_original = yiq_original[:, :, 0].ravel()
    
            # Calcular canal Y de la imagen procesada
            yiq_procesada = mf.rgb_a_yiq(self.imagen_procesada)
            y_procesada = yiq_procesada[:, :, 0].ravel()

            mf.mostrar_histograma(self, y_original, y_procesada)
           

def main():
    ventana = tk.Tk()
    AppFiltrosYIQ(ventana)
    ventana.mainloop()


if __name__ == "__main__":
    main()
