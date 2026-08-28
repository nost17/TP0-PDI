from collections.abc import Callable
import numpy as np

type ImagenArray = np.ndarray[np.dtype[np.float64]]

MATRIZ_SEPIA = np.array(
    [
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ]
)


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


def sepia(imagen: ImagenArray, mezcla: float = 1):
    # im.reshape((-1, 3)) "aplana" la imagen: en vez de (alto, ancho, 3), queda
    # una lista larga de píxeles, cada uno con sus 3 valores R, G, B
    pixeles_planos = imagen.reshape((-1, 3))

    # Multiplicamos cada píxel (fila) por la matriz elegida (transformación de color)
    # y volvemos a darle la forma original de la imagen con .reshape(imagen.shape)
    transformada = (pixeles_planos @ MATRIZ_SEPIA.T).reshape(imagen.shape)

    # np.clip() recorta los valores para que queden siempre entre 0 y 1
    # (algunas matrices, como sepia, pueden generar valores fuera de ese rango)
    transformada = np.clip(transformada, 0, 1)

    # "mezcla" va de 0.0 a 1.0: interpolamos linealmente entre la imagen original
    # y la transformada, según ese porcentaje
    return (1 - mezcla) * imagen + mezcla * transformada


FILTROS: dict[str, Callable[[ImagenArray], ImagenArray]] = {
    "Sin filtro": None,
    "Normal": original,
    "Escala de grises": escala_grises,
    "Sepia": sepia,
    "Invertido": invertido,
    "Canal Rojo": solo_canal(0),
    "Canal Verde": solo_canal(1),
    "Canal Azul": solo_canal(2),
}
