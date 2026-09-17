from typing import Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

type ImagenArray = np.ndarray[np.dtype[np.float64]]

MAT_YIQ = np.array(
    [
        [0.299, 0.595716, 0.211456],
        [0.587, -0.274453, -0.522591],
        [0.114, -0.321263, 0.311135],
    ]
)


def rgb_a_yiq(_im):
    _rgb = _im.reshape((-1, 3))
    _yiq = _rgb @ MAT_YIQ
    _yiq = _yiq.reshape(_im.shape)
    return _yiq


def yiq_a_rgb(_im):
    return np.clip(
        (_im.reshape((-1, 3)) @ np.linalg.inv(MAT_YIQ)).reshape(_im.shape), 0.0, 1.0
    )


def clamp(img: ImagenArray) -> ImagenArray:
    return np.clip(img, 0.0, 1.0)


def division_segura(img1: ImagenArray, img2: ImagenArray) -> ImagenArray:
    return clamp(np.divide(img1, img2 + 1e-7))


def interpolacion(img1, img2):
    img3: ImagenArray = np.zeros(img1.shape)
    YA, IA, QA = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
    YB, IB, QB = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]
    suma_y = YA + YB
    img3[:, :, 1] = (YA * IA + YB * IB) / suma_y
    img3[:, :, 2] = (YA * QA + YB * QB) / suma_y
    return img3, suma_y


def igualar_dimensiones(img1, img2):
    min_alto = min(img1.shape[0], img2.shape[0])
    min_ancho = min(img1.shape[1], img2.shape[1])

    def recortar_centro(img, alto_obj, ancho_obj):
        alto, ancho = img.shape[:2]
        inicio_y = (alto - alto_obj) // 2
        inicio_x = (ancho - ancho_obj) // 2

        return img[inicio_y : inicio_y + alto_obj, inicio_x : inicio_x + ancho_obj]

    img1_rec = recortar_centro(img1, min_alto, min_ancho)
    img2_rec = recortar_centro(img2, min_alto, min_ancho)

    return img1_rec, img2_rec


class Aritmetica:
    __imgA_rgb: ImagenArray
    __imgB_rgb: ImagenArray
    __imgA_yiq: ImagenArray
    __imgB_yiq: ImagenArray

    def __init__(self, imgA: ImagenArray, imgB: ImagenArray):
        self.imgA = imgA
        self.imgB = imgB

    @property
    def imgA(self) -> ImagenArray:
        return self.__imgA_rgb

    @property
    def imgB(self) -> ImagenArray:
        return self.__imgB_rgb

    @property
    def imgA_yiq(self) -> ImagenArray:
        return self.__imgA_yiq

    @property
    def imgB_yiq(self) -> ImagenArray:
        return self.__imgB_yiq

    @imgA.setter
    def imgA(self, nuevo_img: ImagenArray) -> None:
        self.__imgA_rgb = nuevo_img
        self.__imgA_yiq = rgb_a_yiq(nuevo_img)

    @imgB.setter
    def imgB(self, nuevo_img: ImagenArray) -> None:
        self.__imgB_rgb = nuevo_img
        self.__imgB_yiq = rgb_a_yiq(nuevo_img)

    def _sumar_yiq(self, clampear=True) -> ImagenArray:
        img3, suma_y = interpolacion(self.imgA_yiq, self.imgB_yiq)
        if clampear:
            img3[:, :, 0] = clamp(suma_y)
        else:
            img3[:, :, 0] = suma_y / 2.0
        return yiq_a_rgb(img3)

    def _restar_yiq(self, clampear=True, absoluto=False):
        img3, _ = interpolacion(self.imgA_yiq, self.imgB_yiq)
        resta_y = self.imgA_yiq[:, :, 0] - self.imgB_yiq[:, :, 0]
        if absoluto:
            img3[:, :, 0] = abs(resta_y)
        elif clampear:
            img3[:, :, 0] = clamp(resta_y)
        else:
            img3[:, :, 0] = (resta_y + 1.0) / 2.0
        return yiq_a_rgb(img3)

    def sumar(self, yiq=False, clampear=True) -> ImagenArray:
        suma: ImagenArray = None
        if yiq:
            suma = self._sumar_yiq(clampear)
        else:
            suma = self.imgA + self.imgB
            suma = clamp(suma) if clampear else suma / 2.0
        return suma

    def restar(self, yiq=False, clampear=True, absoluto=False) -> ImagenArray:
        resta: ImagenArray = None
        if yiq:
            resta = self._restar_yiq(clampear, absoluto)  # BUGFIX interno
        else:
            resta = self.imgA - self.imgB
            if absoluto:
                resta = abs(resta)
            elif clampear:
                resta = clamp(resta)
            else:
                resta = resta / 2.0
        return resta

    def producto(self, yiq=False) -> ImagenArray:
        resultado: ImagenArray = None
        if yiq:
            resultado, _ = interpolacion(self.imgA_yiq, self.imgB_yiq)
            resultado[:, :, 0] = clamp(self.imgA_yiq[:, :, 0] * self.imgB_yiq[:, :, 0])
            resultado = yiq_a_rgb(resultado)
        else:
            resultado = clamp(self.imgA * self.imgB)
        return resultado

    def cociente(self, yiq=False) -> ImagenArray:
        resultado: ImagenArray = None
        if yiq:
            resultado, _ = interpolacion(self.imgA_yiq, self.imgB_yiq)
            resultado[:, :, 0] = division_segura(
                self.imgA_yiq[:, :, 0], self.imgB_yiq[:, :, 0]
            )
            resultado = yiq_a_rgb(resultado)
        else:
            resultado = division_segura(self.imgA, self.imgB)
        return resultado

    def if_yiq(self, darker=False) -> ImagenArray:
        Y1 = self.imgA_yiq[:, :, 0]
        Y2 = self.imgB_yiq[:, :, 0]
        condicion = (Y1 < Y2) if darker else (Y1 > Y2)
        img3 = np.where(
            np.expand_dims(condicion, axis=-1), self.imgA_yiq, self.imgB_yiq
        )
        return yiq_a_rgb(img3)


class MainApp:
    def __init__(self, ventana: tk.Tk):
        self.ventana = ventana
        self.ventana.title("Aritmetica de Pixeles - PDI")
        self.ventana.geometry("1150x500")
        ttk.Style().theme_use("xpnative")

        self.img1_array = None
        self.img2_array = None
        self.img_resultado_array = None

        self.operaciones: dict[str, dict[str, callable]] = {
            "Suma": {
                "RGB clampeado": lambda ar: ar.sumar(yiq=False, clampear=True),
                "RGB promedio": lambda ar: ar.sumar(yiq=False, clampear=False),
                "YIQ clampeado": lambda ar: ar.sumar(yiq=True, clampear=True),
                "YIQ promedio": lambda ar: ar.sumar(yiq=True, clampear=False),
            },
            "Resta": {
                "RGB clampeado": lambda ar: ar.restar(
                    yiq=False, clampear=True, absoluto=False
                ),
                "RGB promedio": lambda ar: ar.restar(
                    yiq=False, clampear=False, absoluto=False
                ),
                "RGB absoluto": lambda ar: ar.restar(
                    yiq=False, clampear=False, absoluto=True
                ),
                "YIQ clampeado": lambda ar: ar.restar(
                    yiq=True, clampear=True, absoluto=False
                ),
                "YIQ promedio": lambda ar: ar.restar(
                    yiq=True, clampear=False, absoluto=False
                ),
                "YIQ absoluto": lambda ar: ar.restar(
                    yiq=True, clampear=False, absoluto=True
                ),
            },
            "Producto": {
                "RGB": lambda ar: ar.producto(yiq=False),
                "YIQ": lambda ar: ar.producto(yiq=True),
            },
            "Cociente": {
                "RGB": lambda ar: ar.cociente(yiq=False),
                "YIQ": lambda ar: ar.cociente(yiq=True),
            },
            "If": {
                "lighter": lambda ar: ar.if_yiq(darker=False),
                "darker": lambda ar: ar.if_yiq(darker=True),
            },
        }

        self.crear_interfaz()

    def crear_interfaz(self):
        frame_imgs = ttk.Frame(self.ventana)
        frame_imgs.pack(pady=10, fill=tk.BOTH, expand=True)

        labels: list[ttk.Label] = []
        for i in range(3):
            f = tk.Frame(frame_imgs, width=350, height=350)
            f.pack_propagate(False)
            f.pack(side=tk.LEFT, padx=15, expand=True)

            lbl = ttk.Label(
                f, text="nada jeje", anchor="center", background="lightgray"
            )
            lbl.pack(fill=tk.BOTH, expand=True)
            labels.append(lbl)

        self.lbl_img1 = labels[0]
        self.lbl_img2 = labels[1]
        self.lbl_res = labels[2]

        # 2. Contenedor de los botones
        frame_btns = ttk.Frame(self.ventana)
        frame_btns.pack(pady=15)

        ttk.Button(
            frame_btns, text="Abrir imagen 1", command=lambda: self.abrir_imagen(1)
        ).pack(side=tk.LEFT, padx=25)
        ttk.Button(
            frame_btns, text="Abrir imagen 2", command=lambda: self.abrir_imagen(2)
        ).pack(side=tk.LEFT, padx=25)
        ttk.Button(frame_btns, text="Guardar", command=self.guardar_imagen).pack(
            side=tk.LEFT, padx=25
        )
        ttk.Button(frame_btns, text="Procesar", command=self.procesar).pack(
            side=tk.LEFT, padx=25
        )
        ttk.Button(frame_btns, text="Salir", command=self.ventana.quit).pack(
            side=tk.LEFT, padx=25
        )

        frame_opciones = ttk.Frame(self.ventana)
        frame_opciones.pack(pady=10)

        frame_op = ttk.Frame(frame_opciones)
        frame_op.pack(side=tk.LEFT, padx=40)
        self.combo_op = ttk.Combobox(
            frame_op, values=list(self.operaciones.keys()), state="readonly"
        )
        self.combo_op.current(0)
        self.combo_op.pack()
        ttk.Label(frame_op, text="Operación", font=("Arial", 10, "bold")).pack()
        self.combo_op.bind("<<ComboboxSelected>>", self.actualizar_formatos)

        frame_fmt = ttk.Frame(frame_opciones)
        frame_fmt.pack(side=tk.LEFT, padx=40)
        self.combo_fmt = ttk.Combobox(
            frame_fmt,
            values=list(self.operaciones["Suma"].keys()),
            state="readonly",
            width=15,
        )
        self.combo_fmt.current(0)
        self.combo_fmt.pack()
        ttk.Label(frame_fmt, text="Formato", font=("Arial", 10, "bold")).pack()

    def actualizar_formatos(self, event=None):
        operacion_sel: str = self.combo_op.get()
        formatos: list[str] = list(self.operaciones[operacion_sel].keys())
        self.combo_fmt.config(values=formatos)
        self.combo_fmt.current(0)

    def abrir_imagen(self, numero):
        ruta: str = filedialog.askopenfilename(
            title=f"Seleccionar imagen {numero}",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if not ruta:
            return

        imagen_pil = Image.open(ruta).convert("RGB")
        imagen_normalizada: ImagenArray = np.array(imagen_pil) / 255.0

        if numero == 1:
            self.img1_array = imagen_normalizada
            self.mostrar_imagen(self.img1_array, self.lbl_img1)
        else:
            self.img2_array = imagen_normalizada
            self.mostrar_imagen(self.img2_array, self.lbl_img2)

    def mostrar_imagen(self, array_img, label_widget):
        img_pil = Image.fromarray((np.clip(array_img, 0, 1) * 255).astype(np.uint8))
        img_pil.thumbnail((350, 350))
        img_tk = ImageTk.PhotoImage(img_pil)

        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk

    def procesar(self):
        if self.img1_array is None or self.img2_array is None:
            messagebox.showwarning("Atención", "Falta cargar una de las imágenes.")
            return

        img1_rec, img2_rec = igualar_dimensiones(self.img1_array, self.img2_array)

        self.mostrar_imagen(img1_rec, self.lbl_img1)
        self.mostrar_imagen(img2_rec, self.lbl_img2)

        ar = Aritmetica(img1_rec, img2_rec)

        op: str = self.combo_op.get()
        fmt: str = self.combo_fmt.get()

        if op in self.operaciones and fmt in self.operaciones[op]:
            funcion_matematica = self.operaciones[op][fmt]
            res = funcion_matematica(ar)

            self.img_resultado_array = res
            self.mostrar_imagen(res, self.lbl_res)

    def guardar_imagen(self):
        if self.img_resultado_array is not None:
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                initialfile="imagen_procesada.png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            )
            if path:
                img_pil = Image.fromarray(
                    (np.clip(self.img_resultado_array, 0, 1) * 255).astype(np.uint8)
                )
                img_pil.save(path)
                # messagebox.showinfo("Éxito", "Imagen guardada correctamente.")
        else:
            messagebox.showwarning(
                "Atención", "No hay ninguna imagen procesada para guardar."
            )


def main():
    ventana = tk.Tk()
    app = MainApp(ventana)
    ventana.mainloop()


if __name__ == "__main__":
    main()
