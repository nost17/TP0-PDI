from collections.abc import Callable
import numpy as np

type ImagenArray = np.ndarray[np.dtype[np.float64]]


def escala_grises(imagen: ImagenArray) -> ImagenArray:
    """
    Convierte una imagen RGB a escala de grises.

    La imagen llega como array:
        alto x ancho x 3

    El promedio de los tres canales genera un único valor
    de intensidad.

    Después repetimos ese canal 3 veces para conservar
    el formato RGB.
    """

    gris = imagen.mean(axis=2)

    resultado = np.stack([gris, gris, gris], axis=2)

    return resultado


def _obtener_canal(imagen: ImagenArray, canal: int) -> ImagenArray:
    """
    Conserva solamente un canal RGB.

    canal:
        0 -> R
        1 -> G
        2 -> B
    """

    resultado = np.zeros_like(imagen)

    resultado[:, :, canal] = imagen[:, :, canal]

    return resultado


def solo_canal(canal: int) -> Callable[[ImagenArray], ImagenArray]:
    def _canal(imagen: ImagenArray) -> ImagenArray:
        return _obtener_canal(imagen, canal)

    return _canal


def invertido(imagen: ImagenArray) -> ImagenArray:
    invertido = (np.clip(imagen, 0, 1) * 255).astype(np.uint8)
    invertido = 255 - invertido
    return np.array(invertido) / 255.0


def original(imagen: ImagenArray) -> ImagenArray:
    return imagen.copy()


FILTROS: dict[str, Callable[[ImagenArray], ImagenArray]] = {
    "Original": original,
    "Escala de grises": escala_grises,
    "Invertido": invertido,
    "Canal Rojo": solo_canal(0),
    "Canal Verde": solo_canal(1),
    "Canal Azul": solo_canal(2),
}
