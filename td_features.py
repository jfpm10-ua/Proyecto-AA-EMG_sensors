import numpy as np
import pandas as pd

# time-domain feature extraction (explicar por qué elegí estas y explicarlas)


df = pd.read_csv("dataset.csv") # dataset completo
df = df.drop(df.columns[-1], axis=1) # eliminamos label

def td_features_dataset(df: pd.DataFrame):
    """
    Convierte el dataset (sin label) a métricas del dominio temporal:
    RMS, WL, MAV, SSC, ZC.

    Args:
        df(pd.DataFrame): Dataframe de entrada, púramente numérico.

    Returns:
        df(pd.DataFrame): DataFrame de métricas de dominio temporal.
    """

    data = np.array(df)
    n_f = data.shape[0]
    # divimos en bloques de sensores (8 sensores x 8 mediciones)
    data = data.reshape(n_f, 8, 8)

    # rms
    rms = np.sqrt((np.mean(data**2, axis=2)))

    # mav
    mav = np.mean(np.abs(data), axis=2)

    # waveform-length

    # ssc

    # zero-crossing

    # escalar por si tienen magnitudes muy diferentes

    return None # retornar pd.dataframe concatenado [RMS, MAV, WL, SSC, ZC]


if __name__ == "__main__":
    td_features_dataset(df)