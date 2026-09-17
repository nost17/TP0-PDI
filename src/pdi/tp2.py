import tkinter as tk
from tkinter import ttk
import numpy as np

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
    img3 = np.zeros(img1.shape)
    # componentes de la imagen 1
    YA, IA, QA = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
    # componentes de la imagen 2
    YB, IB, QB = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]
    suma_y = YA + YB
    img3[:, :, 1] = (YA * IA + YB * IB) / suma_y
    img3[:, :, 2] = (YA * QA + YB * QB) / suma_y
    return img3, suma_y


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

    @imgA.setter
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
            resta = self._sumar_yiq(clampear)
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
            resultado[:, :, 0] = clamp(self.imgA_yiq[:, :, 0], self.imgB_yiq[:, :, 0])
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
            np.expand_dims(condicion, axis=-1), self.imgA_yiq, self.imgA_yiq
        )
        return yiq_a_rgb(img3)


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
