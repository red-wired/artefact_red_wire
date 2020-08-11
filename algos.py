
# METHODE COLOMBE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import pywt

def detection(data, intervalle, seuil) :
    db = data.copy()
    artefact = []
    ttotal = []
    ind_j = []
    n = len(db)
    j = 1

    db["HR_mean"] = db["HR"]
    db["HR_median"] = db["HR"]

    for indice in range(1, intervalle-1) :

        if ((np.abs(db.HR[indice] - db.HR[indice+1]) > seuil) or (np.abs(db.HR[indice] - db.HR[indice-1]) > seuil)):
            artefact.append(db.HR[indice])
            ttotal.append(db.time[indice])
            ind_j.append(indice)
            #Calcul de la moyenne mobile pour corriger les artefacts
            HR_mean = np.mean(db.HR_mean[(indice):(indice + intervalle)])
            db.HR_mean[indice] = HR_mean
            #Calcul de la médiane mobile pour corriger les artefacts
            HR_median = np.median(db.HR_median[(indice):(indice + intervalle)])
            db.HR_median[indice] = HR_median
            j = j + 1

    for indice in range(intervalle, n-intervalle):
        
        if ((np.abs(db.HR[indice] - db.HR[indice+1]) > seuil) or (np.abs(db.HR[indice] - db.HR[indice-1]) > seuil)):
            artefact.append(db.HR[indice])
            ttotal.append(db.time[indice])
            ind_j.append(indice)
            #Calcul de la moyenne mobile pour corriger les artefacts
            HR_mean = np.mean(db.HR_mean[(indice - intervalle):(indice + intervalle)])
            db.HR_mean[indice] = HR_mean
            #Calcul de la médiane mobile pour corriger les artefacts
            HR_median = np.median(db.HR_median[(indice - intervalle):(indice + intervalle)])
            db.HR_median[indice] = HR_median
            j = j + 1
            
        elif ((np.abs(db.HR[indice] - db.HR[indice-1]) == 0) and (np.abs(db.HR_mean[indice] - db.HR_mean[indice-1]) != 0)):
                        
            #Correction avec la moyenne
            HR_mean = db.HR_mean[indice - 1]
            db.HR_mean[indice] = HR_mean

        elif ((np.abs(db.HR[indice] - db.HR[indice-1]) == 0) and (np.abs(db.HR_median[indice] - db.HR_median[indice-1]) != 0)):
                        
            #Correction avec la médiane
            HR_median = db.HR_median[indice - 1]
            db.HR_median[indice] = HR_median
            
        elif ((np.abs(db.HR[indice] - db.HR_mean[indice-1]) > 5) and (np.abs(db.HR_mean[indice] - db.HR_mean[indice-1]) != 0)):
            
            #Calcul de la moyenne mobile pour corriger les artefacts
            HR_mean = np.mean(db.HR_mean[(indice - intervalle):(indice + intervalle)])
            db.HR_mean[indice] = HR_mean
            
        elif ((np.abs(db.HR[indice] - db.HR_median[indice-1]) > 5) and (np.abs(db.HR_median[indice] - db.HR_median[indice-1]) != 0)):

            #Calcul de la médiane mobile pour corriger les artefacts
            HR_median = np.median(db.HR_median[(indice - intervalle):(indice + intervalle)])
            db.HR_median[indice] = HR_median
            
    for indice in range(n-intervalle-2, n-1) :
        
        if ((np.abs(db.HR[indice] - db.HR[indice+1]) > seuil) or (np.abs(db.HR[indice] - db.HR[indice-1]) > seuil)):
            
            artefact.append(db.HR[indice])
            ttotal.append(db.time[indice])
            ind_j.append(indice)
            #Calcul de la moyenne mobile pour corriger les artefacts
            HR_mean = np.mean(db.HR_mean[(indice):(indice + intervalle)])
            db.HR_mean[indice] = HR_mean
            #Calcul de la médiane mobile pour corriger les artefacts
            HR_median = np.median(db.HR_median[(indice):(indice + intervalle)])
            db.HR_median[indice] = HR_median
            j = j + 1
            
    return db

def method1(df, intervalle):
    seuil = 30
    return detection(df, intervalle, seuil)

# ===============================================================================================

# METHODE REEM

def lissage(data,lisserValue):
    db = data.copy()
    db['HR_modified'] = db.HR.rolling(lisserValue, min_periods=1).mean()
    return db

def method2(df, intervalle):
    return lissage(df, intervalle)

# ===============================================================================================

# METHODE CLEMENT-ULYSSE

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret/n

def simplification1(df, k, n):
    data = df.copy()

    median = data['HR'].median() # ++ mise en place de médianes locales
    data = pd.DataFrame(data['HR'].replace(0.0,median))
    data['HR'].fillna(median, inplace=True)

    data['a1'] = moving_average(np.array(data['HR']), n=n)*(1+k)
    data['a2'] = moving_average(np.array(data['HR']), n=n)*(1-k)

    data['HR-method3'] = data['HR']
    data['HR-method3'][0] = median
    for i in range(1, len(data)):
        if data['HR'][i] < data['a2'][i] :
            data['HR-method3'][i] = data['a2'][i]
        elif data['HR'][i] > data['a1'][i] :
            data['HR-method3'][i] = data['a1'][i]
    return data

def method3(df):
    k = 0.00133 # la freq card baisse de 40% max en 300s = 0.00133 /s
    n = 5
    return simplification1(df, k, n)


def lowpassfilter(df, signal, threshold, thresh_mode, wavelet_mode, wavedec_mode, waverec_mode):
    data = df.copy()
    threshold = threshold*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet_mode, mode=wavedec_mode)
    coeff[1:] = (pywt.threshold(i, value=threshold, mode=thresh_mode) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet_mode, mode=waverec_mode)
    data['HR-method4'] = pd.DataFrame(reconstructed_signal, columns= {'HR-method4'})['HR-method4']
    return data

def method4(df, threshold_slider, threshold_mode, wavelet_mode, waverdec_mode):
    wavedec_mode = waverdec_mode
    waverec_mode = waverdec_mode
    signal = df.HR.fillna(np.mean(df.HR)).values # Toutes les valeurs doivent exister
    rec = lowpassfilter(df, signal, threshold_slider, threshold_mode, wavelet_mode, wavedec_mode, waverec_mode)
    return rec
