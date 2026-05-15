import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# time-domain feature extraction (por qué elegí estas y explicarlas)

def td_features_dataset(df: pd.DataFrame, scaler=None):
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
    wl = np.sum(np.abs(np.diff(data, axis=2)), axis=2)

    # ssc (n_f, 8, 6) la fórmula es de i=2 a N-1
    x_diff_ant = data[:,:,1:-1] - data[:,:,:-2]
    x_diff_post = data[:,:,1:-1] - data[:,:,2:]
    ssc = np.sum((x_diff_ant*x_diff_post) < 0, axis=2)

    # zero-crossing
    threshold = 0.01 
    product = data[:, :, :-1] * data[:, :, 1:]
    diff_abs = np.abs(data[:, :, 1:] - data[:, :, :-1]) 
    zc_condition = (product < 0) & (diff_abs >= threshold)
    zc = np.sum(zc_condition, axis=2)

    # concatenado [RMS, MAV, WL, SSC, ZC]
    result = np.concatenate((rms, mav, wl, ssc, zc), axis=1)
    if scaler is not None:
        result = scaler.fit_transform(result)
    # escalar por si tienen magnitudes muy diferentes
    return pd.DataFrame(result)


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv") # dataset completo
    df = df.drop(df.columns[-1], axis=1) # eliminamos label
    df_td = td_features_dataset(df)