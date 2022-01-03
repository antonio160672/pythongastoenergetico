import sys

sys.path.append("../sensormotion")
sys.path.append("../gastoenergetico")
from sensormotion.pa import *
from sensormotion.signal import *
from crate import client
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gastoenergetico.recuperacion import *
from gastoenergetico.dataformat import *
from gastoenergetico.filter_counts import *

df = recuperacionData("holamundo")
(
    cinturaejesx,
    cinturaejesy,
    cinturaejesz,
    manoejesx,
    manoejesy,
    manoejesz,
    piernaejesx,
    piernaejesy,
    piernaejesz,
    time,
    sampling_rate,
) = formatData(df)


# plt.scatter(time,x)
#
print(len(time))
print(len(cinturaejesx))
conjunto = [
    time,
    cinturaejesx,
    cinturaejesy,
    cinturaejesz,
    manoejesx,
    manoejesy,
    manoejesz,
    piernaejesx,
    piernaejesy,
    piernaejesz,
]
s = pd.DataFrame(conjunto).T
s.columns = [
    "time",
    "cinturaejesx",
    "cinturaejesy",
    "cinturaejesz",
    "piernaejesx",
    "piernaejesy",
    "piernaejesz",
    "manoejesx",
    "manoejesy",
    "manoejesz",
]
# s.to_csv('actividades.csv', header=True, index=False)
epoca = 10
(
    cinturaejesx_counts,
    cinturaejesy_counts,
    cinturaejesz_counts,
    manoejesx_counts,
    manoejesy_counts,
    manoejesz_counts,
    piernaejesx_counts,
    piernaejesy_counts,
    piernaejesz_counts,
    cinturaejesx_f2_counts,
    cinturaejesy_f2_counts,
    cinturaejesz_f2_counts,
    manoejesx_f2_counts,
    manoejesy_f2_counts,
    manoejesz_f2_counts,
    piernaejesx_f2_counts,
    piernaejesy_f2_counts,
    piernaejesz_f2_counts,
) = filter_counts(
    0.8,
    2.5,
    epoca,
    time,
    sampling_rate,
    cinturaejesx,
    cinturaejesy,
    cinturaejesz,
    manoejesx,
    manoejesy,
    manoejesz,
    piernaejesx,
    piernaejesy,
    piernaejesz,
)


if cinturaejesx.size > 0:
    cinturavm = vector_magnitude(
        cinturaejesx_counts, cinturaejesy_counts, cinturaejesz_counts
    )
if manoejesx.size > 0:
    manovm = vector_magnitude(manoejesx_counts, manoejesy_counts, manoejesz_counts)
if piernaejesx.size > 0:
    piernavm = vector_magnitude(
        piernaejesx_counts, piernaejesy_counts, piernaejesz_counts
    )

if cinturaejesx.size > 0:
    cinturaFvm = vector_magnitude(
        cinturaejesx_f2_counts, cinturaejesy_f2_counts, cinturaejesz_f2_counts
    )
if manoejesx.size > 0:
    manoFvm = vector_magnitude(
        manoejesx_f2_counts, manoejesy_f2_counts, manoejesz_f2_counts
    )
if piernaejesx.size > 0:
    piernaFvm = vector_magnitude(
        piernaejesx_f2_counts, piernaejesy_f2_counts, piernaejesz_f2_counts
    )

print("Valores de la cintura \n")
print(cinturaejesx_f2_counts)
print(cinturaejesy_f2_counts)
print(cinturaejesz_f2_counts)
print("\n Valores de la mano \n")
print(manoejesx_f2_counts)
print(manoejesy_f2_counts)
print(manoejesz_f2_counts)
print("\n Valores de la pierna \n")
print(piernaejesx_f2_counts)
print(piernaejesy_f2_counts)
print(piernaejesz_f2_counts)


if "cinturavm" in globals():
    print("cintura sin filtro: \n", cinturavm)
if "manovm" in globals():
    print("mano sin filtro: \n", manovm)
if "piernavm" in globals():
    print("pierna sin filtro: \n", piernavm)


if "cinturaFvm" in globals():
    print("\ncintura con filtro: \n", cinturaFvm)
if "manoFvm" in globals():
    print("mano con filtro: \n", manoFvm)
if "piernaFvm" in globals():
    print("pierna con filtro: \n", piernaFvm)
