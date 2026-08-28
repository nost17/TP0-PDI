import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from .filtros import ImagenArray, FILTROS

ESPACIO: dict[str, int | tuple[int]] = {
    "normal": 10,
    "izq": (10, 0),
    "der": (0, 10),
    "eizq": (5, 0),
    "eder": (0, 5),
}


class MainApp:
    ventana: tk.Tk

    def __init__(self, ventana: tk.Tk):
        self.ventana = ventana
        self.ventana.title("PDI - Tkinter básico")
        self.ventana.geometry("1000x700")

        self.imagen_normal = None
        self.imagen_actual = None
        self.imagen_original = None

        self.crear_interfaz()

    def iniciar(self) -> None:
        self.ventana.mainloop()

    def crear_interfaz(self) -> None:
        panel = ttk.Frame(self.ventana)
        panel.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10,
        )

        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1, uniform="paneles")
        panel.columnconfigure(1, weight=1, uniform="paneles")

        panel_izq = ttk.Frame(panel)
        panel_izq.grid(row=0, column=0, sticky=tk.NSEW, padx=ESPACIO["der"])

        panel_izq.columnconfigure(0, weight=1)
        panel_izq.rowconfigure(0, weight=2, uniform="filas")
        panel_izq.rowconfigure(1, weight=3, uniform="filas")

        cont_histo = ttk.Frame(panel_izq)
        cont_histo.grid(row=0, column=0, sticky=tk.NSEW, pady=ESPACIO["der"])
        cont_histo.pack_propagate(False)

        self.label_histo: ttk.Label = self._crear_visual(cont_histo, "Histograma")
        self.label_histo.pack(fill=tk.BOTH, expand=True)

        cont_filtro = ttk.Frame(panel_izq)
        cont_filtro.grid(row=1, column=0, sticky=tk.NSEW)  # Eliminamos el rowspan=2
        cont_filtro.pack_propagate(False)

        self.label_img_filtro: ttk.Label = self._crear_visual(cont_filtro, "Filtro")
        self.label_img_filtro.pack(fill=tk.BOTH, expand=True)

        panel_der = ttk.Frame(panel)
        panel_der.grid(row=0, column=1, sticky=tk.NSEW)

        self._crear_botones(panel_der).pack(side=tk.TOP, fill=tk.X)

        self.label_img_normal: ttk.Label = self._crear_visual(
            panel_der,
            "Abrí una imagen",
        )

        self.label_img_normal.pack(fill=tk.BOTH, expand=True, pady=ESPACIO["izq"])

    def _crear_botones(self, raiz: ttk.Frame) -> ttk.Frame:
        nombres_filtros: tuple[str] = tuple(FILTROS.keys())

        self.operacion = tk.StringVar(value=nombres_filtros[0])

        barra = ttk.Frame(raiz)

        btn_menu_filtros = tk.OptionMenu(barra, self.operacion, *nombres_filtros)

        btn_menu_filtros.config(width=max(len(nombre) for nombre in nombres_filtros))

        btn_abrir_imagen = ttk.Button(
            barra,
            text="Abrir imagen",
            command=self.abrir_imagen,
            padding=(6, 3),
        )
        btn_aplicar_filtro = ttk.Button(
            barra,
            text="Aplicar filtro",
            command=self.aplicar_filtro,
            padding=(6, 3),
        )
        btn_guardar = ttk.Button(
            barra,
            text="Guardar",
            command=self.guardar_imagen,
            padding=(6, 3),
        )
        btn_intercambiar = ttk.Button(
            barra,
            text="Intercambio",
            command=self.intercambiar_imagenes,
            padding=(6, 3),
        )

        for i, b in enumerate(
            [
                btn_abrir_imagen,
                btn_menu_filtros,
                btn_aplicar_filtro,
                btn_guardar,
                btn_intercambiar,
            ]
        ):
            b.grid(row=0, column=i)
            barra.columnconfigure(i, weight=1)

        return barra

    def _crear_visual(self, raiz: ttk.Frame, texto) -> ttk.Label:
        return ttk.Label(
            raiz, text=texto, anchor=tk.CENTER, borderwidth=1, relief=tk.SOLID
        )

    def aplicar_filtro(self) -> None:

        if self.imagen_normal is None:
            messagebox.showwarning("Atención", "Primero abrí una imagen.")
            return

        operacion = self.operacion.get()

        filtro = FILTROS[operacion]

        resultado: ImagenArray = self.imagen_original if filtro is None else filtro(self.imagen_normal)

        self.imagen_actual = resultado

        self.mostrar_imagen(resultado, self.label_img_filtro)

    def abrir_imagen(self):

        ruta: str = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )

        if not ruta:
            return

        # Pillow abre la imagen.
        imagen_pil: Image = Image.open(ruta).convert("RGB")

        # Convertimos Pillow -> NumPy.
        #
        # /255.0 normaliza:
        #
        # 0   -> 0.0
        # 255 -> 1.0
        #
        self.imagen_normal: ImagenArray = np.array(imagen_pil) / 255.0

        # La imagen actual comienza siendo igual a la original.
        self.imagen_actual: ImagenArray = self.imagen_normal.copy()

        self.imagen_original: ImagenArray = self.imagen_normal.copy()

        self.mostrar_imagen(self.imagen_actual, self.label_img_normal)
        self.aplicar_filtro()

    def mostrar_imagen(
        self, array_imagen: ImagenArray, contenedor: ttk.Label, altura=None
    ) -> None:

        if array_imagen is None:
            self.label_img_filtro.foto = None
            return

        # Convertimos 0-1 nuevamente a 0-255.
        imagen_uint8 = (np.clip(array_imagen, 0, 1) * 255).astype(np.uint8)

        # NumPy -> Pillow
        imagen_pil: Image = Image.fromarray(imagen_uint8)

        # Acá calculamos el ancho y alto del contenedor
        # para que la imagen ocupe el mismo tamaño y quede bien
        ancho: int = contenedor.winfo_width()
        alto: int = contenedor.winfo_height()

        if ancho <= 1 or alto <= 1:
            ancho, alto = 600, 600

        imagen_pil.thumbnail((ancho, alto), Image.Resampling.LANCZOS)

        # Pillow -> Tkinter
        foto = ImageTk.PhotoImage(imagen_pil)

        # Guardamos la referencia.
        contenedor.foto = foto

        # Mostramos la imagen.
        contenedor.config(image=foto, text="")

    def guardar_imagen(self):
        if self.imagen_actual is not None:
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                initialfile="imagen_procesada.png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            )
            if path:
                img: Image = Image.fromarray(
                    (np.clip(self.imagen_actual, 0, 1) * 255).astype(np.uint8)
                )
                img.save(path)

    def intercambiar_imagenes(self) -> None:
        if self.imagen_actual is None or self.imagen_normal is None:
            return

        self.imagen_normal = self.imagen_actual.copy()
        self.imagen_actual = None

        self.mostrar_imagen(self.imagen_normal, self.label_img_normal)
        self.mostrar_imagen(self.imagen_normal, self.label_img_filtro)


def main() -> None:
    app = MainApp(tk.Tk())
    app.iniciar()


if __name__ == "__main__":
    main()
