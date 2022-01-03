import sys
sys.path.append("../sensormotion")
from sensormotion.pa import *
from sensormotion.signal import *
from crate import client
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def asiganacion(Df1, Df2, Df3):
    for i in range(0, len(Df1)):
        if not Df1[i]:
            if not Df1[i] and not Df2[i] and Df3:
                Df1[i] = Df3[i]
            else:
                Df1[i] = Df2[i]
    return Df1


def recuperacionData(args):
    
    connection = client.connect(
        "http://187.188.90.137:4200/", username="crate", timeout=5)
    cursor = connection.cursor()
    consulta = '1637731212000'
    #cadena="SELECT entity_id, pierna, mano, cintura, cinturaejesx, cinturaejesy, cinturaejesz, piernaejesx, piernaejesy, piernaejesz, manoejesx, manoejesy, manoejesz,fecha_inicio ,fecha_fin FROM doc.etpersona  where fecha_inicio =?"
    #cursor.execute("SELECT name FROM locations WHERE name = ?", ("Algol"))
    cadena = "SELECT entity_id,cintura, pierna, mano,  cinturaejesx, cinturaejesy, cinturaejesz, piernaejesx, piernaejesy, piernaejesz, manoejesx, manoejesy, manoejesz,fecha_inicio ,fecha_fin FROM doc.etpersona where entity_id='AntonioTrotar1'  order by fecha_inicio"
    cursor.execute(cadena)

    cabecera = [column[0] for column in cursor.description]
    result = cursor.fetchall()
    largo = len(result)
    df = pd.DataFrame(result)
    df.columns = cabecera
    # df=df.drop(5)
    #df  = df.dropna()
    dfaux = pd.DataFrame()
    i = 1
    while i <= largo:
        FeIni = df.iloc[i]['fecha_inicio']
        FeFin = df.iloc[i]['fecha_fin']
        indiceini = df.index[df.fecha_inicio == FeIni].values
        indicefin = df.index[df.fecha_fin == FeFin].values
        if len(indiceini) == 3:
            data = pd.DataFrame(asiganacion(df.iloc[indiceini[0]].tolist(
            ), df.iloc[indiceini[1]].tolist(), df.iloc[indiceini[2]].tolist())).T
            dfaux = dfaux.append(data, ignore_index=True)
            i = i+3
            continue
        if len(indicefin) == 3:
            data = pd.DataFrame(asiganacion(df.iloc[indicefin[0]].tolist(
            ), df.iloc[indicefin[1]].tolist(), df.iloc[indicefin[2]].tolist())).T
            dfaux = dfaux.append(data, ignore_index=True)
            i = i+3
            continue
        if len(indiceini) == 2:
            data = pd.DataFrame(asiganacion(
                df.iloc[indiceini[0]].tolist(), df.iloc[indiceini[1]].tolist(), "")).T
            dfaux = dfaux.append(data, ignore_index=True)
            i = i+2
            continue
        if len(indicefin) == 2:
            data = pd.DataFrame(asiganacion(
                df.iloc[indicefin[0]].tolist(), df.iloc[indicefin[1]].tolist(), "")).T
            dfaux = dfaux.append(data, ignore_index=True)
            i = i+2
            continue
    dfaux.columns = cabecera
    df = dfaux
    #df.to_csv('actividades.csv', header=True, index=False)
    display(df)
    return df