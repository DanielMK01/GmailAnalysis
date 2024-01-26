from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def kmeans(arr, iscsv):
    data = arr
    if (iscsv):
        data = np.delete(arr, 0, 0)
        data = np.delete(data, 0, 1)
    kmean = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(data)
    return kmean

def csv_to_arr(filename):
    data = np.genfromtxt('{}.csv'.format(filename), delimiter=',', dtype=None, encoding=None)
    return data

def df_to_np(df):
    nparray = df.to_numpy()
    return nparray

def getTable(arr, labels, iscsv):
    data = arr
    if (iscsv):
        data = np.delete(arr, 0, 0)
        data = np.delete(data, 0, 1)
    data = np.concatenate((data, labels), axis=1)
    return data