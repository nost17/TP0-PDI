# 1)Implementar un aplicativo que levante una imagen en nivel de gris (o que la convierta a YIQ y retenga solo el Y)
# y aplique el filtrado por convolución aquí visto:

# -------Filtros lineales: Pasabajos, Pasabanda, Pasaaltos y Detectores de bordes-----------
# a) Pasabajos: Plano y Bartlett 3x3, 5x5, 7x7. Gaussiano 5x5 y 7x7.
# b) Detectores de bordes: Laplaciano v4 y v8. Sobel las 8 orientaciones.
# c)Pasabanda, pasaaltos de frecuencia de corte 0,2 y 0,4.
#
# 2) Respecto del "cierre" de la imagen cerca de los bordes, implementar
# la técnica no matemáticamente perfecta de "repetir" artificialmente filas o columnas cuando estas hagan falta.
# Por ejemplo, si el kernel es 3x3 y estoy en la primera columna, y no hay ninguna columna a la izquierda, entonces
# repito la primera columna como si estuviese en la imagen original.

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


def crear_kernel_plano(tamano: int) -> ImagenArray:
    return np.ones((tamano, tamano), dtype=np.float64) / (tamano * tamano)


# sigma determina que tan "ancha" es la campana gaussiana, y por lo tanto que tan fuerte es el filtrado.
def crear_kernel_gaussiano(tamano: int, sigma: float) -> ImagenArray:
    ax = np.arange(-tamano // 2 + 1, tamano // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def crear_kernel_impulso(tamano: int) -> ImagenArray:
    kernel = np.zeros((tamano, tamano), dtype=np.float64)
    kernel[tamano // 2, tamano // 2] = 1.0
    return kernel


def _convolution(image, kernel=np.ones((1, 1))) -> ImagenArray:
    convolved = np.zeros((np.array(image.shape) - np.array(kernel.shape) + 1))
    for x in range(convolved.shape[0]):
        for y in range(convolved.shape[1]):
            convolved[x, y] = (
                image[x : x + kernel.shape[0], y : y + kernel.shape[1]] * kernel
            ).sum()
    return convolved


# Matrices para Bartlett
barlett_3X3: ImagenArray = (
    np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
)

barlett_5X5: ImagenArray = (
    np.array(
        [
            [1, 2, 3, 2, 1],
            [2, 4, 6, 4, 2],
            [3, 6, 9, 6, 3],
            [2, 4, 6, 4, 2],
            [1, 2, 3, 2, 1],
        ],
        dtype=np.float64,
    )
    / 81.0
)

barlett_7X7: ImagenArray = (
    np.array(
        [
            [1, 2, 3, 4, 3, 2, 1],
            [2, 4, 6, 8, 6, 4, 2],
            [3, 6, 9, 12, 9, 6, 3],
            [4, 8, 12, 16, 12, 8, 4],
            [3, 6, 9, 12, 9, 6, 3],
            [2, 4, 6, 8, 6, 4, 2],
            [1, 2, 3, 4, 3, 2, 1],
        ],
        dtype=np.float64,
    )
    / 256.0
)


FILTROS: dict[str, Callable[[ImagenArray], ImagenArray]] = {
    # a) Pasabajos
    "Pasabajos Llano 3x3": lambda: crear_kernel_plano(3),
    "Pasabajos Llano 5x5": lambda: crear_kernel_plano(5),
    "Pasabajos Llano 7x7": lambda: crear_kernel_plano(7),
    "Pasabajos Bartlett 3x3": lambda: barlett_3X3,
    "Pasabajos Bartlett 5x5": lambda: barlett_5X5,
    "Pasabajos Bartlett 7x7": lambda: barlett_7X7,
    "Pasabajos Gaussiano 5x5": lambda: crear_kernel_gaussiano(5, 1.0),
    "Pasabajos Gaussiano 7x7": lambda: crear_kernel_gaussiano(7, 1.5),
    # b) Detectores de bordes
    "Laplaciano v4": lambda: np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64
    ),
    "Laplaciano v8": lambda: np.array(
        [[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64
    ),
    # Sobel 8 orientaciones
    "Sobel N (Norte)": lambda: np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64
    ),
    "Sobel NE (Noreste)": lambda: np.array(
        [[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float64
    ),
    "Sobel E (Este)": lambda: np.array(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64
    ),
    "Sobel SE (Sureste)": lambda: np.array(
        [[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float64
    ),
    "Sobel S (Sur)": lambda: np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64
    ),
    "Sobel SO (Suroeste)": lambda: np.array(
        [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float64
    ),
    "Sobel O (Oeste)": lambda: np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64
    ),
    "Sobel NO (Noroeste)": lambda: np.array(
        [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float64
    ),
    # c) Pasabanda y Pasaaltos
    "Pasaaltos (frec = 0.2)": lambda: (
        crear_kernel_impulso(15) - crear_kernel_gaussiano(15, 1.0 / (2 * np.pi * 0.2))
    ),
    "Pasaaltos (frec = 0.4)": lambda: (
        crear_kernel_impulso(15) - crear_kernel_gaussiano(15, 1.0 / (2 * np.pi * 0.4))
    ),
    "Pasabanda (frec = 0.2 - 0.4)": lambda: (
        crear_kernel_gaussiano(15, 1.0 / (2 * np.pi * 0.4))
        - crear_kernel_gaussiano(15, 1.0 / (2 * np.pi * 0.2))
    ),
}


# 2) se implementa la técnica de "repetir" artificialmente filas o columnas cuando estas hagan falta, usando padding 'edge' en la función aplicar_convolucion.
def aplicar_convolucion(y: ImagenArray, kernel: np.ndarray) -> ImagenArray:
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    y_padded = np.pad(y, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")

    resultado = _convolution(y_padded, kernel)
    return resultado


class AppFiltros:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("PDI TP4 - Convolucion")
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
            controles, text="Filtro de Convolución", font=("Arial", 12, "bold")
        ).pack(pady=10)

        self.filtro_var = tk.StringVar()
        self.combo_filtros = ttk.Combobox(
            controles,
            textvariable=self.filtro_var,
            values=list(FILTROS.keys()),
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

        if nombre_filtro in FILTROS:
            kernel = FILTROS[nombre_filtro]()
            y_filtrado = aplicar_convolucion(yiq_mod[:, :, 0], kernel)
            yiq_mod[:, :, 0] = y_filtrado

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


def main():
    ventana = tk.Tk()
    AppFiltros(ventana)
    ventana.mainloop()


if __name__ == "__main__":
    main()
