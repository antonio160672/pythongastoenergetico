from sensormotion.pa import *
from sensormotion.signal import *
from crate import client
from IPython.display import display
import sys
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


def downsample(array, npts):
    from scipy.interpolate import interp1d
    interpolated = interp1d(np.arange(len(array)), array,
                            axis=0, fill_value='extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled


sys.path.append("../sensormotion")

connection = client.connect(
    "http://187.188.90.137:4200/", username="crate", timeout=5)
cursor = connection.cursor()
consulta = '1637731212000'
#cadena="SELECT entity_id, pierna, mano, cintura, cinturaejesx, cinturaejesy, cinturaejesz, piernaejesx, piernaejesy, piernaejesz, manoejesx, manoejesy, manoejesz,fecha_inicio ,fecha_fin FROM doc.etpersona  where fecha_inicio =?"
#cursor.execute("SELECT name FROM locations WHERE name = ?", ("Algol"))
cadena = "SELECT entity_id,cintura, pierna, mano,  cinturaejesx, cinturaejesy, cinturaejesz, piernaejesx, piernaejesy, piernaejesz, manoejesx, manoejesy, manoejesz,fecha_inicio ,fecha_fin FROM doc.etpersona where entity_id='ExperimentoReposoAntonio'  order by fecha_inicio"
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


# plt.scatter(time,x)
#
print(len(time))
print(len(cinturaejesx))
conjunto = [time, cinturaejesx, cinturaejesy, cinturaejesz, manoejesx,
            manoejesy, manoejesz, piernaejesx, piernaejesy, piernaejesz]
s = pd.DataFrame(conjunto).T
s.columns = ['time', 'cinturaejesx',	'cinturaejesy',	'cinturaejesz',	'piernaejesx',
             'piernaejesy',	'piernaejesz',	'manoejesx',	'manoejesy',	'manoejesz']
#s.to_csv('actividades.csv', header=True, index=False)


b, a = build_filter((0.2, 2.5), sampling_rate, 'bandpass', filter_order=4)
#b, a = build_filter(10, sampling_rate, 'low', filter_order=4)


if cinturaejesx.size > 0:
    cinturaejesx_f2 = filter_signal(b, a, cinturaejesx)
    cinturaejesy_f2 = filter_signal(b, a, cinturaejesy)
    cinturaejesz_f2 = filter_signal(b, a, cinturaejesz)

if manoejesx.size > 0:
    manoejesx_f2 = filter_signal(b, a, manoejesx)
    manoejesy_f2 = filter_signal(b, a, manoejesy)
    manoejesz_f2 = filter_signal(b, a, manoejesz)

if piernaejesx.size > 0:
    piernaejesx_f2 = filter_signal(b, a, piernaejesx)
    piernaejesy_f2 = filter_signal(b, a, piernaejesy)
    piernaejesz_f2 = filter_signal(b, a, piernaejesz)

epoca = 10
if cinturaejesx.size > 0:
    cinturaejesx_counts = convert_counts(
        cinturaejesx, time, epoch=epoca, rectify='full', integrate='trapezoid', plot=False)
    cinturaejesy_counts = convert_counts(
        cinturaejesy, time, epoch=epoca, rectify='full', integrate='trapezoid', plot=False)
    cinturaejesz_counts = convert_counts(
        cinturaejesz, time, epoch=epoca, rectify='full', integrate='trapezoid', plot=False)

if manoejesx.size > 0:
    manoejesx_counts = convert_counts(
        manoejesx, time, epoch=epoca, rectify='full', integrate='trapezoid', plot=False)
    manoejesy_counts = convert_counts(
        manoejesy, time, epoch=epoca, rectify='full', integrate='trapezoid', plot=False)
    manoejesz_counts = convert_counts(
        manoejesz, time, epoch=epoca, rectify='full', integrate='trapezoid', plot=False)

if piernaejesx.size > 0:
    piernaejesx_counts = convert_counts(
        piernaejesx, time, epoch=epoca, rectify='full', integrate='trapezoid', plot=False)
    piernaejesy_counts = convert_counts(
        piernaejesy, time, epoch=epoca, rectify='full', integrate='trapezoid', plot=False)
    piernaejesz_counts = convert_counts(
        piernaejesz, time, epoch=epoca, rectify='full', integrate='trapezoid', plot=False)

if cinturaejesx.size > 0:
    cinturaejesx_f2_counts = convert_counts(
        cinturaejesx_f2, time, epoch=epoca, rectify='full', integrate='simpson', plot=False)
    cinturaejesy_f2_counts = convert_counts(
        cinturaejesy_f2, time, epoch=epoca, rectify='full', integrate='simpson', plot=False)
    cinturaejesz_f2_counts = convert_counts(
        cinturaejesz_f2, time, epoch=epoca, rectify='full', integrate='simpson', plot=False)
if manoejesx.size > 0:
    manoejesx_f2_counts = convert_counts(
        manoejesx_f2, time, time_scale='ms', epoch=epoca, rectify='full', integrate='simpson', plot=False)
    manoejesy_f2_counts = convert_counts(
        manoejesy_f2, time, time_scale='ms', epoch=epoca, rectify='full', integrate='simpson', plot=False)
    manoejesz_f2_counts = convert_counts(
        manoejesz_f2, time, time_scale='ms', epoch=epoca, rectify='full', integrate='simpson', plot=False)
if piernaejesx.size > 0:
    piernaejesx_f2_counts = convert_counts(
        piernaejesx_f2, time, time_scale='ms', epoch=epoca, rectify='full', integrate='simpson', plot=False)
    piernaejesy_f2_counts = convert_counts(
        piernaejesy_f2, time, time_scale='ms', epoch=epoca, rectify='full', integrate='simpson', plot=False)
    piernaejesz_f2_counts = convert_counts(
        piernaejesz_f2, time, time_scale='ms', epoch=epoca, rectify='full', integrate='simpson', plot=False)

if cinturaejesx.size > 0:
    cinturavm = vector_magnitude(
        cinturaejesx_counts, cinturaejesy_counts, cinturaejesz_counts)
if manoejesx.size > 0:
    manovm = vector_magnitude(
        manoejesx_counts, manoejesy_counts, manoejesz_counts)
if piernaejesx.size > 0:
    piernavm = vector_magnitude(
        piernaejesx_counts, piernaejesy_counts, piernaejesz_counts)

if cinturaejesx.size > 0:
    cinturaFvm = vector_magnitude(
        cinturaejesx_f2_counts, cinturaejesy_f2_counts, cinturaejesz_f2_counts)
if manoejesx.size > 0:
    manoFvm = vector_magnitude(
        manoejesx_f2_counts, manoejesy_f2_counts, manoejesz_f2_counts)
if piernaejesx.size > 0:
    piernaFvm = vector_magnitude(
        piernaejesx_f2_counts, piernaejesy_f2_counts, piernaejesz_f2_counts)


print("cintura sin filtro: \n", cinturavm)
print("mano sin filtro: \n", manovm)
print("pierna sin filtro: \n", piernavm)
print("suma de vectores: \n", sum(cinturavm), sum(manovm), sum(piernavm))

print("\ncintura con filtro: \n", cinturaFvm)
print("mano con filtro: \n", manoFvm)
print("pierna con filtro: \n", piernaFvm)
print("suma de vectores: \n", sum(cinturaFvm), sum(manoFvm), sum(piernaFvm))
