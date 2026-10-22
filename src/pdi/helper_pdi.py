import numpy as np
from PIL import Image, ImageTk

type ImagenArray = np.ndarray[np.dtype[np.float64]]

# region Conversion RGB - YIQ

_MAT_YIQ = np.array(
    [
        [0.299, 0.595716, 0.211456],
        [0.587, -0.274453, -0.522591],
        [0.114, -0.321263, 0.311135],
    ]
)


def rgb_a_yiq(_im: ImagenArray) -> ImagenArray:
    _rgb = _im.reshape((-1, 3))
    _yiq = _rgb @ _MAT_YIQ
    return _yiq.reshape(_im.shape)


def yiq_a_rgb(_im: ImagenArray) -> ImagenArray:
    return np.clip(
        (_im.reshape((-1, 3)) @ np.linalg.inv(_MAT_YIQ)).reshape(_im.shape), 0.0, 1.0
    )


# endregion

# region Aritmetica de pixeles

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


class AritImg:
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

    @staticmethod
    def _sumar_yiq(imgA_yiq, imgB_yiq, clampear=True) -> ImagenArray:
        img3, suma_y = interpolacion(imgA_yiq, imgB_yiq)
        if clampear:
            img3[:, :, 0] = clamp(suma_y)
        else:
            img3[:, :, 0] = suma_y / 2.0
        return yiq_a_rgb(img3)

    @staticmethod
    def _restar_yiq(
        imgA_yiq, imgB_yiq, clampear=True, absoluto=False
    ) -> ImagenArray:
        img3, _ = interpolacion(imgA_yiq, imgB_yiq)
        resta_y = imgA_yiq[:, :, 0] - imgB_yiq[:, :, 0]
        if absoluto:
            img3[:, :, 0] = abs(resta_y)
        elif clampear:
            img3[:, :, 0] = clamp(resta_y)
        else:
            img3[:, :, 0] = (resta_y + 1.0) / 2.0
        return yiq_a_rgb(img3)

    @staticmethod
    def sumar_img(
        imgA, imgB, imgA_yiq=None, imgB_yiq=None, yiq=False, clampear=True
    ) -> ImagenArray:
        if yiq:
            return AritImg._sumar_yiq(imgA_yiq, imgB_yiq, clampear)
        suma = imgA + imgB
        return clamp(suma) if clampear else suma / 2.0

    @staticmethod
    def restar_img(
        imgA,
        imgB,
        imgA_yiq=None,
        imgB_yiq=None,
        yiq=False,
        clampear=True,
        absoluto=False,
    ) -> ImagenArray:
        if yiq:
            return AritImg._restar_yiq(
                imgA_yiq, imgB_yiq, clampear, absoluto
            )
        resta = imgA - imgB
        if absoluto:
            return abs(resta)
        elif clampear:
            return clamp(resta)
        return resta / 2.0

    @staticmethod
    def producto_img(
        imgA, imgB, imgA_yiq=None, imgB_yiq=None, yiq=False
    ) -> ImagenArray:
        if yiq:
            resultado, _ = interpolacion(imgA_yiq, imgB_yiq)
            resultado[:, :, 0] = clamp(imgA_yiq[:, :, 0] * imgB_yiq[:, :, 0])
            return yiq_a_rgb(resultado)
        return clamp(imgA * imgB)

    @staticmethod
    def cociente_img(
        imgA, imgB, imgA_yiq=None, imgB_yiq=None, yiq=False
    ) -> ImagenArray:
        if yiq:
            resultado, _ = interpolacion(imgA_yiq, imgB_yiq)
            resultado[:, :, 0] = division_segura(imgA_yiq[:, :, 0], imgB_yiq[:, :, 0])
            return yiq_a_rgb(resultado)
        return division_segura(imgA, imgB)

    @staticmethod
    def if_yiq_img(imgA_yiq, imgB_yiq, darker=False) -> ImagenArray:
        Y1 = imgA_yiq[:, :, 0]
        Y2 = imgB_yiq[:, :, 0]
        condicion = (Y1 < Y2) if darker else (Y1 > Y2)
        img3 = np.where(np.expand_dims(condicion, axis=-1), imgA_yiq, imgB_yiq)
        return yiq_a_rgb(img3)

    def sumar(self, yiq=False, clampear=True) -> ImagenArray:
        return self.sumar_img(
            self.imgA, self.imgB, self.imgA_yiq, self.imgB_yiq, yiq, clampear
        )

    def restar(self, yiq=False, clampear=True, absoluto=False) -> ImagenArray:
        return self.restar_img(
            self.imgA, self.imgB, self.imgA_yiq, self.imgB_yiq, yiq, clampear, absoluto
        )

    def producto(self, yiq=False) -> ImagenArray:
        return self.producto_img(
            self.imgA, self.imgB, self.imgA_yiq, self.imgB_yiq, yiq
        )

    def cociente(self, yiq=False) -> ImagenArray:
        return self.cociente_img(
            self.imgA, self.imgB, self.imgA_yiq, self.imgB_yiq, yiq
        )

    def if_yiq(self, darker=False) -> ImagenArray:
        return self.if_yiq_img(self.imgA_yiq, self.imgB_yiq, darker)

# endregion

# region Utilidades para interfaz gráfica

def img_a_pil(array_imagen: np.ndarray) -> Image:
    imagen_uint8 = (clamp(array_imagen) * 255).astype(np.uint8)
    return Image.fromarray(imagen_uint8)

def img_a_widget(array_imagen: np.ndarray, tamano=(500, 500)) -> ImageTk.PhotoImage:
    imagen_pil: Image = img_a_pil(array_imagen)
    imagen_pil.thumbnail(tamano)
    return ImageTk.PhotoImage(imagen_pil)

# endregion

# region Filtros

def filtro_raiz(img: ImagenArray) -> ImagenArray:
    return np.sqrt(clamp(img))


def filtro_cuadrado(img: ImagenArray) -> ImagenArray:
    return np.square(clamp(img))

# endregion

def mi_funcion():
    print("HOLA")
