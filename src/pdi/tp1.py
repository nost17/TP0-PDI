import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

MAT_YIQ = np.array(
    [
        [0.299, 0.595716, 0.211456],
        [0.587, -0.274453, -0.522591],
        [0.114, -0.321263, 0.311135],
    ]
)

I_MAX: float = 0.5957
I_MIN: float = -0.5957
Q_MAX: float = 0.5226
Q_MIN: float = -0.5226


def rgb_a_yiq(_im):
    _rgb = _im.reshape((-1, 3))
    _yiq = _rgb @ MAT_YIQ
    _yiq = _yiq.reshape(_im.shape)
    return _yiq


def yiq_a_rgb(_im):
    return np.clip(
        (_im.reshape((-1, 3)) @ np.linalg.inv(MAT_YIQ)).reshape(_im.shape), 0.0, 1.0
    )


def clamp(imagen, minimo: float = 0, maximo: float = 1):
    return np.clip(imagen, minimo, maximo)


def procesar_yiq(imagen, alpha, beta):
    yiq = rgb_a_yiq(imagen)
    yiq_mod = yiq.copy()
    yiq_mod[:, :, 0] = clamp(yiq_mod[:, :, 0] * alpha)
    yiq_mod[:, :, 1] = clamp(yiq_mod[:, :, 1] * beta, I_MIN, I_MAX)
    yiq_mod[:, :, 2] = clamp(yiq_mod[:, :, 2] * beta, Q_MIN, Q_MAX)
    rgb_mod = yiq_a_rgb(yiq_mod)
    return rgb_mod, yiq_mod


def mostrar_canal_yiq(imagen_yiq, canal):
    c = np.zeros_like(imagen_yiq)
    if canal != 0:
        c[:,:,0] = 0.4
    c[:, :, canal] = imagen_yiq[:, :, canal]
    return yiq_a_rgb(c)


class AppYIQ:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("PDI - Transformación YIQ")
        self.ventana.geometry("1300x600")
        self.imagen_original = None
        self.imagen_actual = None
        self.imagen_yiq = None
        self.imagen_rgb_mod = None
        self.crear_interfaz()

    def crear_interfaz(self):
        panel = tk.Frame(self.ventana)
        panel.pack(fill="both", expand=True, padx=10, pady=10)

        frame_imagenes = tk.Frame(panel)
        frame_imagenes.pack(side="left", fill="both", expand=True, padx=10)

        self.label_img_izq = tk.Label(
            frame_imagenes, text="Imagen Original", bg="#dddddd"
        )
        self.label_img_izq.pack(side="left", fill="both", expand=True, padx=5)

        self.label_img_der = tk.Label(
            frame_imagenes, text="Imagen Modificada", bg="#dddddd"
        )
        self.label_img_der.pack(side="left", fill="both", expand=True, padx=5)

        controles = tk.Frame(panel)
        controles.pack(side="right", fill="y", padx=10)

        tk.Label(controles, text="Transformación YIQ", font=("Arial", 14, "bold")).pack(
            pady=10
        )

        self.valor_alpha = tk.DoubleVar(value=1.0)
        self.valor_beta = tk.DoubleVar(value=1.0)

        self._crear_slider(controles, "a (Luminancia)", self.valor_alpha)
        self._crear_slider(controles, "b (Saturación)", self.valor_beta)

        tk.Label(controles, text="Vista", font=("Arial", 11, "bold")).pack(
            anchor="w", pady=(20, 0)
        )

        self.vista = tk.StringVar(value="RGB modificada")

        for opcion in ["RGB modificada", "Canal Y", "Canal I", "Canal Q"]:
            tk.Radiobutton(
                controles,
                text=opcion,
                variable=self.vista,
                value=opcion,
                command=self.actualizar_vista,
            ).pack(anchor="w")

        tk.Button(controles, text="Aplicar", command=self.aplicar).pack(
            fill="x", pady=(20, 5)
        )
        tk.Button(controles, text="Guardar como", command=self.guardar_imagen, relief="ridge").pack(
            fill="x", pady=5
        )
        tk.Button(controles, text="Abrir imagen", command=self.abrir_imagen, relief="ridge").pack(
            fill="x", pady=5
        )

        tk.Button(controles, text="Restaurar", command=self.restaurar, relief="ridge").pack(
            side="bottom", fill="x", pady=5
        )

        self.estado = tk.Label(self.ventana, text="Listo.", anchor="w")
        self.estado.pack(side="bottom", fill="x", padx=10, pady=10)

    def _crear_slider(self, padre, etiqueta, variable):
        tk.Label(padre, text=etiqueta).pack(anchor="w", pady=(10, 0))
        tk.Scale(
            padre,
            variable=variable,
            from_=0.0,
            to=3.0,
            resolution=0.05,
            orient="horizontal",
            length=200,
        ).pack(fill="x")

    def abrir_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if not ruta:
            return

        imagen_pil = Image.open(ruta).convert("RGB")
        self.imagen_original = np.array(imagen_pil) / 255.0
        self.imagen_actual = self.imagen_original.copy()
        self.imagen_yiq = rgb_a_yiq(self.imagen_actual)
        self.imagen_rgb_mod = self.imagen_actual.copy()

        self.mostrar_imagen(self.imagen_original, self.label_img_izq)
        self.mostrar_imagen(self.imagen_actual, self.label_img_der)
        self.estado.config(text="Imagen cargada correctamente.")

    def aplicar(self):
        if self.imagen_original is None:
            messagebox.showwarning("Atención", "Primero abrí una imagen.")
            return

        fY: float = self.valor_alpha.get()
        fIQ: float = self.valor_beta.get()

        rgb_mod, yiq_mod = procesar_yiq(self.imagen_original, fY, fIQ)

        self.imagen_rgb_mod = rgb_mod
        self.imagen_yiq = yiq_mod

        self.actualizar_vista()
        self.estado.config(text=f"Factores aplicados: Y*{fY:.1f}  IQ*{fIQ:.1f}")

    def actualizar_vista(self):
        if self.imagen_yiq is None:
            return

        vista: str = self.vista.get()

        if vista == "RGB modificada":
            self.imagen_actual = self.imagen_rgb_mod
        elif vista == "Canal Y":
            self.imagen_actual = mostrar_canal_yiq(self.imagen_yiq, 0)
        elif vista == "Canal I":
            self.imagen_actual = mostrar_canal_yiq(self.imagen_yiq, 1)
        elif vista == "Canal Q":
            self.imagen_actual = mostrar_canal_yiq(self.imagen_yiq, 2)

        self.mostrar_imagen(self.imagen_actual, self.label_img_der)
        self.estado.config(text=f"Vista actual: {vista}")

    def mostrar_imagen(self, array_imagen, label_widget):
        imagen_uint8 = (np.clip(array_imagen, 0, 1) * 255).astype(np.uint8)
        imagen_pil = Image.fromarray(imagen_uint8)
        imagen_pil.thumbnail((500, 500))
        foto = ImageTk.PhotoImage(imagen_pil)
        label_widget.foto = foto
        label_widget.config(image=foto, text="")

    def restaurar(self):
        if self.imagen_original is None:
            return

        self.imagen_actual = self.imagen_original.copy()
        self.imagen_yiq = None
        self.imagen_rgb_mod = None

        self.valor_alpha.set(1.0)
        self.valor_beta.set(1.0)
        self.vista.set("RGB modificada")

        self.mostrar_imagen(self.imagen_actual, self.label_img_der)
        self.estado.config(text="Imagen original restaurada.")

    def guardar_imagen(self):
        if self.imagen_actual is not None:
            path = filedialog.asksaveasfilename(
                # defaultextension=".png",
                initialfile="imagen_transformacion_yiq.png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tiff"), ("BMP", "*.bmp")],
            )
            if path:
                img_pil = Image.fromarray(
                    (np.clip(self.imagen_actual, 0, 1) * 255).astype(np.uint8)
                )
                img_pil.save(path)
                # messagebox.showinfo("Éxito", "Imagen guardada correctamente.")
        else:
            messagebox.showwarning(
                "Atención", "No hay ninguna imagen procesada para guardar."
            )

def main():
    ventana = tk.Tk()
    AppYIQ(ventana)
    ventana.mainloop()


if __name__ == "__main__":
    main()