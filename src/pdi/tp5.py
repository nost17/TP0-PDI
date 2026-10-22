
from collections.abc import Callable
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

type ImagenArray = np.ndarray[np.dtype[np.float64]]

MAT_YIQ = np.array(
    [
        [0.299, 0.595716, 0.211456],
        [0.587, -0.274453, -0.522591],
        [0.114, -0.321263, 0.311135],
    ]
)


def rgb_a_yiq(_im: ImagenArray) -> ImagenArray:
    _rgb = _im.reshape((-1, 3))
    _yiq = _rgb @ MAT_YIQ
    return _yiq.reshape(_im.shape)


def yiq_a_rgb(_im: ImagenArray) -> ImagenArray:
    return np.clip(
        (_im.reshape((-1, 3)) @ np.linalg.inv(MAT_YIQ)).reshape(_im.shape), 0.0, 1.0
    )


def aplicar_operacion_base( image: ImagenArray, operacion: str, tam_kernel: int = 3) -> ImagenArray:
    pad = tam_kernel // 2
    image_padded = np.pad(image, ((pad, pad), (pad,pad)), mode="edge")
    resultado = np.zeros_like(image)
    for x in range (resultado.shape[0]):
        for y in range (resultado.shape[1]):
            ventana = image_padded[x : x + tam_kernel, y : y + tam_kernel]
            if operacion == 'erosion':
                resultado[x,y] = np.min(ventana)
            elif operacion == 'dilatacion':
                resultado[x,y] = np.max(ventana)
            elif operacion == 'mediana':
                resultado[x,y] = np.median(ventana)
    return resultado

def aplicar_morfologia(image: ImagenArray, tipo_filtro: str) -> ImagenArray:

    if tipo_filtro == "Erosión 3x3":
        return aplicar_operacion_base(image, 'erosion')
    elif tipo_filtro == "Dilatación 3x3":
        return aplicar_operacion_base(image, 'dilatacion')
    elif tipo_filtro == "Mediana 3x3":
        return aplicar_operacion_base(image, 'mediana')

    elif tipo_filtro == "Apertura 3x3":
        erosionado = aplicar_operacion_base(image,'erosion')
        return aplicar_operacion_base(erosionado,'dilatacion')
    elif tipo_filtro == "Cierre 3x3":
        dilatado = aplicar_operacion_base(image,'dilatacion')
        return aplicar_operacion_base(dilatado,'erosion')
    elif tipo_filtro == "Borde Exterior":
        dilatado = aplicar_operacion_base(image,'dilatacion')
        return np.clip(dilatado - image, 0.0, 1.0)
    


class AppFiltros:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("PDI TP5 - Morfología")
        self.ventana.geometry("1200x650")

        self.imagen_original = None
        self.imagen_procesada = None
        self.imagen_yiq_mod = None

        self.crear_interfaz()

    def crear_interfaz(self):
        panel = ttk.Frame(self.ventana)
        panel.pack(fill="both", expand=True, padx=10, pady=10)

        frame_imagenes = ttk.Frame(panel)
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

        ttk.Label(
            controles, text="Morfología", font=("Arial", 12, "bold")
        ).pack(pady=10)

        self.filtro_var = tk.StringVar()
        self.combo_filtros = ttk.Combobox(
            controles,
            textvariable=self.filtro_var,
            values=["Erosión 3x3", "Dilatación 3x3", "Mediana 3x3", "Apertura 3x3", "Cierre 3x3", "Borde Exterior"],
            state="readonly",
            width=30,
        )
        self.combo_filtros.pack(pady=10)
        self.combo_filtros.current(0)

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
        ttk.Button(controles, text="<- Copiar", command=self.copiar_imagen).pack(
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

        yiq = rgb_a_yiq(self.imagen_original)

        yiq_mod = np.zeros_like(yiq)
        yiq_mod[:, :, 0] = yiq[:, :, 0]

        nombre_filtro = self.filtro_var.get()

        y_filtrado = aplicar_morfologia(yiq_mod[:,:,0], nombre_filtro)
        yiq_mod[:,:,0] = y_filtrado

        self.imagen_yiq_mod = yiq_mod
        self.imagen_procesada = yiq_a_rgb(yiq_mod)

        self.mostrar_imagen(self.imagen_procesada, self.label_procesada)
        self.estado.config(text=f"Filtro aplicado: {nombre_filtro}")

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
            initialfile="imagen_con_filtro.png",
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

    def copiar_imagen(self):
        if self.imagen_procesada is None:
            messagebox.showwarning("Atención", "No hay imagen procesada para copiar")
            return

        self.imagen_original = self.imagen_procesada.copy()

        self.mostrar_imagen(self.imagen_original, self.label_original)
        self.estado.config(text="Imagen copiada")

def main():
    ventana = tk.Tk()
    AppFiltros(ventana)
    ventana.mainloop()


if __name__ == "__main__":
    main()
