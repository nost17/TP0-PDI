import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pdi.helper_pdi import ImagenArray
import pdi.helper_pdi as helper


def aplicar_operacion_base(
    image: ImagenArray, operacion: str, tam_kernel: int = 3
) -> ImagenArray:
    pad = tam_kernel // 2
    image_padded = np.pad(image, ((pad, pad), (pad, pad)), mode="edge")
    resultado = np.zeros_like(image)
    for x in range(resultado.shape[0]):
        for y in range(resultado.shape[1]):
            ventana = image_padded[x : x + tam_kernel, y : y + tam_kernel]
            if operacion == "erosion":
                resultado[x, y] = np.min(ventana)
            elif operacion == "dilatacion":
                resultado[x, y] = np.max(ventana)
            elif operacion == "mediana":
                resultado[x, y] = np.median(ventana)
    return resultado


def aplicar_morfologia(image: ImagenArray, tipo_filtro: str) -> ImagenArray:
    if tipo_filtro == "Erosión 3x3":
        return aplicar_operacion_base(image, "erosion")
    elif tipo_filtro == "Dilatación 3x3":
        return aplicar_operacion_base(image, "dilatacion")
    elif tipo_filtro == "Mediana 3x3":
        return aplicar_operacion_base(image, "mediana")
    elif tipo_filtro == "Apertura 3x3":
        erosionado = aplicar_operacion_base(image, "erosion")
        return aplicar_operacion_base(erosionado, "dilatacion")
    elif tipo_filtro == "Cierre 3x3":
        dilatado = aplicar_operacion_base(image, "dilatacion")
        return aplicar_operacion_base(dilatado, "erosion")
    elif tipo_filtro == "Borde Exterior":
        dilatado = aplicar_operacion_base(image, "dilatacion")
        return helper.clamp(dilatado - image)


class AppFiltros:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("PDI TP5 - Morfología con Histogramas")
        self.ventana.geometry("1400x800")

        self.imagen_original = None
        self.imagen_procesada = None
        self.imagen_yiq_mod = None

        self.crear_interfaz()

    def crear_interfaz(self):
        panel = ttk.Frame(self.ventana)
        panel.pack(fill="both", expand=True, padx=10, pady=10)

        # contenedor grande
        frame_imagenes = ttk.Frame(panel)
        frame_imagenes.pack(side="left", fill="both", expand=True, padx=10)

        # imagen izquierda (original)
        frame_izq = ttk.Frame(frame_imagenes)
        frame_izq.pack(side="left", fill="both", expand=True, padx=5)

        self.label_original = ttk.Label(
            frame_izq, text="Imagen original", anchor="center", background="#dddddd"
        )
        self.label_original.pack(side="top", fill="both", expand=True, pady=5)

        self.fig_orig = Figure(figsize=(2, 2), dpi=100)
        self.ax_orig = self.fig_orig.add_subplot(111)
        self.canvas_orig = FigureCanvasTkAgg(self.fig_orig, master=frame_izq)
        self.canvas_orig.get_tk_widget().pack(side="bottom", fill="both", expand=True)

        # imagen derecha (procesada)
        frame_der = ttk.Frame(frame_imagenes)
        frame_der.pack(side="left", fill="both", expand=True, padx=5)

        self.label_procesada = ttk.Label(
            frame_der, text="Imagen procesada", anchor="center", background="#dddddd"
        )
        self.label_procesada.pack(side="top", fill="both", expand=True, pady=5)

        self.fig_proc = Figure(figsize=(2, 2), dpi=100)
        self.ax_proc = self.fig_proc.add_subplot(111)
        self.canvas_proc = FigureCanvasTkAgg(self.fig_proc, master=frame_der)
        self.canvas_proc.get_tk_widget().pack(side="bottom", fill="both", expand=True)

        controles = ttk.Frame(panel)
        controles.pack(side="right", fill="y", padx=10)

        ttk.Label(controles, text="Morfología", font=("Arial", 12, "bold")).pack(
            pady=10
        )

        self.filtro_var = tk.StringVar()
        self.combo_filtros = ttk.Combobox(
            controles,
            textvariable=self.filtro_var,
            values=[
                "Erosión 3x3",
                "Dilatación 3x3",
                "Mediana 3x3",
                "Apertura 3x3",
                "Cierre 3x3",
                "Borde Exterior",
            ],
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

    def dibujar_histograma(self, imagen_rgb, ax, canvas, titulo):
        ax.clear()
        if imagen_rgb is not None:
            yiq: ImagenArray = helper.rgb_a_yiq(imagen_rgb)
            canal_y = yiq[:, :, 0].flatten()

            # Graficamos
            ax.hist(canal_y, bins=50, range=(0.0, 1.0), color="blue", alpha=0.7)
            ax.set_title(titulo, fontsize=10)
            ax.set_xlim(0, 1)
            ax.tick_params(axis="both", which="major", labelsize=8)

        canvas.draw()

    def abrir_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if not ruta:
            return

        imagen_pil: Image.Image = Image.open(ruta).convert("RGB")
        self.imagen_original = np.array(imagen_pil) / 255.0
        self.imagen_procesada = self.imagen_original.copy()
        self.imagen_yiq_mod = None

        self.mostrar_imagen(self.imagen_original, self.label_original)
        self.mostrar_imagen(self.imagen_procesada, self.label_procesada)

        self.dibujar_histograma(
            self.imagen_original, self.ax_orig, self.canvas_orig, "Histograma Original"
        )
        self.dibujar_histograma(
            self.imagen_procesada,
            self.ax_proc,
            self.canvas_proc,
            "Histograma Procesada",
        )

        self.estado.config(text="Imagen cargada y normalizada a [0,1].")

    def aplicar_filtro(self):
        if self.imagen_original is None:
            messagebox.showwarning("Atención", "Primero abrí una imagen.")
            return

        yiq: ImagenArray = helper.rgb_a_yiq(self.imagen_original)

        yiq_mod = np.zeros_like(yiq)
        yiq_mod[:, :, 0] = yiq[:, :, 0]

        nombre_filtro = self.filtro_var.get()

        y_filtrado = aplicar_morfologia(yiq_mod[:, :, 0], nombre_filtro)
        yiq_mod[:, :, 0] = y_filtrado

        self.imagen_yiq_mod = yiq_mod
        self.imagen_procesada = helper.yiq_a_rgb(yiq_mod)

        self.mostrar_imagen(self.imagen_procesada, self.label_procesada)
        self.dibujar_histograma(
            self.imagen_procesada,
            self.ax_proc,
            self.canvas_proc,
            f"Histograma ({nombre_filtro})",
        )

        self.estado.config(text=f"Filtro aplicado: {nombre_filtro}")

    def mostrar_imagen(self, array_imagen, label_widget):
        foto = helper.img_a_widget(array_imagen)
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

        helper.img_a_pil(self.imagen_procesada).save(ruta)
        self.estado.config(text="Imagen guardada.")

    def restaurar(self):
        if self.imagen_original is None:
            return

        self.imagen_procesada = self.imagen_original.copy()
        self.imagen_yiq_mod = None
        self.mostrar_imagen(self.imagen_procesada, self.label_procesada)
        self.dibujar_histograma(
            self.imagen_procesada,
            self.ax_proc,
            self.canvas_proc,
            "Histograma Procesada",
        )
        self.estado.config(text="Imagen original restaurada.")

    def copiar_imagen(self):
        if self.imagen_procesada is None:
            messagebox.showwarning("Atención", "No hay imagen procesada para copiar")
            return

        self.imagen_original = self.imagen_procesada.copy()

        self.mostrar_imagen(self.imagen_original, self.label_original)
        self.dibujar_histograma(
            self.imagen_original,
            self.ax_orig,
            self.canvas_orig,
            "Histograma Original (Copiada)",
        )
        self.estado.config(text="Imagen copiada")


def main():
    ventana = tk.Tk()
    AppFiltros(ventana)
    ventana.mainloop()


if __name__ == "__main__":
    main()
