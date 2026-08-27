import tkinter as tk


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
        pass


def main() -> None:
    app = MainApp(tk.Tk())
    app.iniciar()


if __name__ == "__main__":
    main()
