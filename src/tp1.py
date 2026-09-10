import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

M_RGB2YIQ = np.array([
    [0.299,  0.595716,  0.211456],
    [0.587, -0.274453, -0.522591],
    [0.114, -0.321263,  0.311135],
])

M_YIQ2RGB = np.linalg.inv(M_RGB2YIQ.T)

def rgb_a_yiq(imagen):
    return np.dot(imagen, M_RGB2YIQ.T)

def yiq_a_rgb(imagen_yiq):
    resultado = np.dot(imagen_yiq, M_YIQ2RGB)
    return np.clip(resultado, 0, 1)

def procesar_yiq(imagen, factor_Y, factor_I, factor_Q):
    yiq = rgb_a_yiq(imagen)
    yiq_mod = yiq.copy()
    yiq_mod[:, :, 0] *= factor_Y
    yiq_mod[:, :, 1] *= factor_I
    yiq_mod[:, :, 2] *= factor_Q
    yiq_mod[:, :, 0] = np.clip(yiq_mod[:, :, 0], 0, 1)
    rgb_mod = yiq_a_rgb(yiq_mod)
    return rgb_mod, yiq_mod

def mostrar_canal_yiq(imagen_yiq, canal):
    c = np.clip(imagen_yiq[:, :, canal], 0, 1)
    return np.stack([c, c, c], axis=2)

class AppYIQ:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("PDI - Transformación YIQ")
        self.ventana.geometry("1000x720")
        self.imagen_original = None
        self.imagen_actual = None
        self.imagen_yiq = None
        self.imagen_rgb_mod = None
        self.crear_interfaz()

    def crear_interfaz(self):
        barra = tk.Frame(self.ventana)
        barra.pack(side="top", fill="x", padx=10, pady=10)

        tk.Button(barra, text="Abrir imagen", command=self.abrir_imagen).pack(side="left", padx=5)
        tk.Button(barra, text="Restaurar", command=self.restaurar).pack(side="left", padx=5)

        panel = tk.Frame(self.ventana)
        panel.pack(fill="both", expand=True, padx=10, pady=10)

        self.label_imagen = tk.Label(panel, text="Abrí una imagen para comenzar", bg="#dddddd")
        self.label_imagen.pack(side="left", fill="both", expand=True, padx=10)

        controles = tk.Frame(panel)
        controles.pack(side="right", fill="y", padx=10)

        tk.Label(controles, text="Transformación YIQ", font=("Arial", 14, "bold")).pack(pady=10)

        self.factor_Y = tk.DoubleVar(value=1.0)
        self.factor_I = tk.DoubleVar(value=1.5)
        self.factor_Q = tk.DoubleVar(value=1.5)

        self._crear_slider(controles, "Factor Y", self.factor_Y)
        self._crear_slider(controles, "Factor I", self.factor_I)
        self._crear_slider(controles, "Factor Q", self.factor_Q)

        tk.Label(controles, text="Vista", font=("Arial", 11, "bold")).pack(anchor="w", pady=(20, 0))

        self.vista = tk.StringVar(value="RGB modificada")

        for opcion in ["RGB modificada", "Canal Y", "Canal I", "Canal Q"]:
            tk.Radiobutton(
                controles,
                text=opcion,
                variable=self.vista,
                value=opcion,
                command=self.actualizar_vista
            ).pack(anchor="w")

        tk.Button(controles, text="Aplicar", command=self.aplicar).pack(fill="x", pady=20)

        self.estado = tk.Label(self.ventana, text="Listo.", anchor="w")
        self.estado.pack(side="bottom", fill="x", padx=10, pady=10)

    def _crear_slider(self, padre, etiqueta, variable):
        tk.Label(padre, text=etiqueta).pack(anchor="w", pady=(10, 0))
        tk.Scale(
            padre, variable=variable, from_=0.0, to=3.0,
            resolution=0.1, orient="horizontal", length=200
        ).pack(fill="x")

    def abrir_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not ruta:
            return

        imagen_pil = Image.open(ruta).convert("RGB")
        self.imagen_original = np.array(imagen_pil) / 255.0
        self.imagen_actual = self.imagen_original.copy()
        self.imagen_yiq = None
        self.imagen_rgb_mod = None

        self.mostrar_imagen(self.imagen_actual)
        self.estado.config(text="Imagen cargada correctamente.")

    def aplicar(self):
        if self.imagen_original is None:
            messagebox.showwarning("Atención", "Primero abrí una imagen.")
            return

        fY = self.factor_Y.get()
        fI = self.factor_I.get()
        fQ = self.factor_Q.get()

        rgb_mod, yiq_mod = procesar_yiq(self.imagen_original, fY, fI, fQ)

        self.imagen_rgb_mod = rgb_mod
        self.imagen_yiq = yiq_mod

        self.actualizar_vista()
        self.estado.config(text=f"Factores aplicados: Y*{fY:.1f}  I*{fI:.1f}  Q*{fQ:.1f}")

    def actualizar_vista(self):
        if self.imagen_yiq is None:
            return

        vista = self.vista.get()

        if vista == "RGB modificada":
            self.imagen_actual = self.imagen_rgb_mod
        elif vista == "Canal Y":
            self.imagen_actual = mostrar_canal_yiq(self.imagen_yiq, 0)
        elif vista == "Canal I":
            self.imagen_actual = mostrar_canal_yiq(self.imagen_yiq, 1)
        elif vista == "Canal Q":
            self.imagen_actual = mostrar_canal_yiq(self.imagen_yiq, 2)

        self.mostrar_imagen(self.imagen_actual)
        self.estado.config(text=f"Vista actual: {vista}")

    def mostrar_imagen(self, array_imagen):
        imagen_uint8 = (np.clip(array_imagen, 0, 1) * 255).astype(np.uint8)
        imagen_pil = Image.fromarray(imagen_uint8)
        imagen_pil.thumbnail((600, 600))
        foto = ImageTk.PhotoImage(imagen_pil)
        self.label_imagen.foto = foto
        self.label_imagen.config(image=foto, text="")

    def restaurar(self):
        if self.imagen_original is None:
            return

        self.imagen_actual = self.imagen_original.copy()
        self.imagen_yiq = None
        self.imagen_rgb_mod = None

        self.factor_Y.set(1.0)
        self.factor_I.set(1.5)
        self.factor_Q.set(1.5)
        self.vista.set("RGB modificada")

        self.mostrar_imagen(self.imagen_actual)
        self.estado.config(text="Imagen original restaurada.")

if __name__ == "__main__":
    ventana = tk.Tk()
    app = AppYIQ(ventana)
    ventana.mainloop()