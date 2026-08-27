import tkinter
import tkinter as tk
from tkinter import ttk

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
        self.ventana.geometry("900x600")
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
        panel_izq.rowconfigure(0, weight=1, uniform="filas")
        panel_izq.rowconfigure(1, weight=3, uniform="filas")

        panel_der = ttk.Frame(panel)
        panel_der.grid(row=0, column=1, sticky=tk.NSEW)

        self._crear_visual(
            panel_izq,
            "Histograma",
        ).grid(row=0, column=0, sticky=tk.NSEW, pady=ESPACIO["der"])
        self._crear_visual(
            panel_izq,
            "Filtro",
        ).grid(row=1, column=0, sticky=tk.NSEW, rowspan=2)

        self._crear_botones(panel_der).pack(side=tk.TOP, fill=tk.X)

        self.label_img_normal: ttk.Label = self._crear_visual(
            panel_der,
            "Abrí una imagen",
        )

        self.label_img_normal.pack(fill=tk.BOTH, expand=True, pady=ESPACIO["izq"])

    def _crear_botones(self, raiz: ttk.Frame) -> ttk.Frame:
        barra = ttk.Frame(raiz)

        btn_abrir_imagen = ttk.Button(
            barra,
            text="Abrir imagen",
            command=lambda: print("> abrir imagen"),
            padding=(6, 3),
        )
        btn_aplicar_filtro = ttk.Button(
            barra,
            text="Aplicar filtro",
            command=lambda: print("> aplicar filtro"),
            padding=(6, 3),
        )
        btn_guardar = ttk.Button(
            barra,
            text="Guardar",
            command=lambda: print("> guardar imagen"),
            padding=(6, 3),
        )
        btn_restaurar = ttk.Button(
            barra,
            text="Restaurar",
            command=lambda: print("> restaurar imagen"),
            padding=(6, 3),
        )

        for i, b in enumerate(
            [btn_abrir_imagen, btn_aplicar_filtro, btn_guardar, btn_restaurar]
        ):
            b.grid(row=0, column=i)

        return barra

    def _crear_visual(self, raiz: ttk.Frame, texto) -> ttk.Label:
        return ttk.Label(
            raiz, text=texto, anchor=tk.CENTER, borderwidth=1, relief=tk.SOLID
        )


def main() -> None:
    app = MainApp(tk.Tk())
    app.iniciar()


if __name__ == "__main__":
    main()
