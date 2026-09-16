import tkinter as tk
from tkinter import ttk


class MainApp:
    ventana: tk.Tk

    def __init__(self, ventana: tk.Tk):
        self.ventana = ventana
        self.ventana.title("PDI - Tkinter básico")
        self.ventana.geometry("1100x800")

        # ('winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative')
        ttk.Style().theme_use("vista")

    def iniciar(self) -> None:
        self.ventana.mainloop()

def main():
    app = MainApp(tk.Tk())
    app.iniciar()


if __name__ == "__main__":
    main()
