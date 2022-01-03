import sys
sys.path.append("../sensormotion")
from sensormotion.pa import *
from sensormotion.signal import *
from crate import client
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def downsample(array, npts):
    from scipy.interpolate import interp1d
    interpolated = interp1d(np.arange(len(array)), array,
                            axis=0, fill_value='extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

def formatData(df):
    cinturaejesx = np.array([])
    cinturaejesy = np.array([])
    cinturaejesz = np.array([])
    piernaejesx = np.array([])
    piernaejesy = np.array([])
    piernaejesz = np.array([])
    manoejesx = np.array([])
    manoejesy = np.array([])
    manoejesz = np.array([])

    contador = 0
    for index, row in df.iterrows():
        for i in range(0, len(row['cinturaejesx'])):
            cinturaejesx = np.append(cinturaejesx, float(row['cinturaejesx'][i]))
            # cinturaejesx.append(float(row['cinturaejesx'][i]))
        for i in range(0, len(row['cinturaejesy'])):
            cinturaejesy = np.append(cinturaejesy, float(row['cinturaejesy'][i]))
            # cinturaejesy.append(float(row['cinturaejesy'][i]))
        for i in range(0, len(row['cinturaejesz'])):
            cinturaejesz = np.append(cinturaejesz, float(row['cinturaejesz'][i]))
            # cinturaejesz.append(float(row['cinturaejesz'][i]))

        for i in range(0, len(row['piernaejesx'])):
            piernaejesx = np.append(piernaejesx, float(row['piernaejesx'][i]))
            # piernaejesx.append(float(row['piernaejesx'][i]))
        for i in range(0, len(row['piernaejesy'])):
            piernaejesy = np.append(piernaejesy, float(row['piernaejesy'][i]))
            # piernaejesy.append(float(row['piernaejesy'][i]))
        for i in range(0, len(row['piernaejesz'])):
            piernaejesz = np.append(piernaejesz, float(row['piernaejesz'][i]))
            # piernaejesz.append(float(row['piernaejesz'][i]))

        for i in range(0, len(row['manoejesx'])):
            manoejesx = np.append(manoejesx, float(row['manoejesx'][i]))
            # manoejesx.append(float(row['manoejesx'][i]))
        for i in range(0, len(row['manoejesy'])):
            manoejesy = np.append(manoejesy, float(row['manoejesy'][i]))
            # manoejesy.append(float(row['manoejesy'][i]))
        for i in range(0, len(row['manoejesz'])):
            manoejesz = np.append(manoejesz, float(row['manoejesz'][i]))
            # manoejesz.append(float(row['manoejesz'][i]))

        contador = contador+1
    print(contador)
    sampling_rate = 102
    segundos = contador*30
    redution = segundos*sampling_rate
    time = np.arange(0, (segundos)*sampling_rate) * 10
    if cinturaejesx.size > 0:
        cinturaejesx = downsample(cinturaejesx, redution)
        cinturaejesy = downsample(cinturaejesy, redution)
        cinturaejesz = downsample(cinturaejesz, redution)
    if manoejesx.size > 0:
        manoejesx = downsample(manoejesx, redution)
        manoejesy = downsample(manoejesy, redution)
        manoejesz = downsample(manoejesz, redution)
    if piernaejesx.size > 0:
        piernaejesx = downsample(piernaejesx, redution)
        piernaejesy = downsample(piernaejesy, redution)
        piernaejesz = downsample(piernaejesz, redution)
    return cinturaejesx,cinturaejesy,cinturaejesz,manoejesx,manoejesy,manoejesz,piernaejesx,piernaejesy,piernaejesz,time,sampling_rate