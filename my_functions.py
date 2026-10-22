import numpy as np
import matplotlib.pyplot as plt

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


def mostrar_histograma(self,original, procesada):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        # Histograma Original
        axes[0].hist(original, bins=256, range=(0.0, 1.0), color="black", alpha=0.7)
        axes[0].set_title("Original")

        # Histograma Procesado
        axes[1].hist(procesada, bins=256, range=(0.0, 1.0), color="blue", alpha=0.7)
        axes[1].set_title("Procesada")

        plt.tight_layout()
        plt.show()