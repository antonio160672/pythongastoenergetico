import sys

sys.path.append("../sensormotion")
from sensormotion.pa import *
from sensormotion.signal import *
from crate import client
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def filter_counts(
    down,
    up,
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
):

    b, a = build_filter((down, up), sampling_rate, "bandpass", filter_order=4)
    # b, a = build_filter(10, sampling_rate, 'low', filter_order=4)

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

    if cinturaejesx.size > 0:
        cinturaejesx_counts = convert_counts(
            cinturaejesx,
            time,
            epoch=epoca,
            rectify="full",
            integrate="trapezoid",
            plot=False,
        )
        cinturaejesy_counts = convert_counts(
            cinturaejesy,
            time,
            epoch=epoca,
            rectify="full",
            integrate="trapezoid",
            plot=False,
        )
        cinturaejesz_counts = convert_counts(
            cinturaejesz,
            time,
            epoch=epoca,
            rectify="full",
            integrate="trapezoid",
            plot=False,
        )

    if manoejesx.size > 0:
        manoejesx_counts = convert_counts(
            manoejesx,
            time,
            epoch=epoca,
            rectify="full",
            integrate="trapezoid",
            plot=False,
        )
        manoejesy_counts = convert_counts(
            manoejesy,
            time,
            epoch=epoca,
            rectify="full",
            integrate="trapezoid",
            plot=False,
        )
        manoejesz_counts = convert_counts(
            manoejesz,
            time,
            epoch=epoca,
            rectify="full",
            integrate="trapezoid",
            plot=False,
        )

    if piernaejesx.size > 0:
        piernaejesx_counts = convert_counts(
            piernaejesx,
            time,
            epoch=epoca,
            rectify="full",
            integrate="trapezoid",
            plot=False,
        )
        piernaejesy_counts = convert_counts(
            piernaejesy,
            time,
            epoch=epoca,
            rectify="full",
            integrate="trapezoid",
            plot=False,
        )
        piernaejesz_counts = convert_counts(
            piernaejesz,
            time,
            epoch=epoca,
            rectify="full",
            integrate="trapezoid",
            plot=False,
        )

    if cinturaejesx.size > 0:
        cinturaejesx_f2_counts = convert_counts(
            cinturaejesx_f2,
            time,
            epoch=epoca,
            rectify="full",
            integrate="simpson",
            plot=False,
        )
        cinturaejesy_f2_counts = convert_counts(
            cinturaejesy_f2,
            time,
            epoch=epoca,
            rectify="full",
            integrate="simpson",
            plot=False,
        )
        cinturaejesz_f2_counts = convert_counts(
            cinturaejesz_f2,
            time,
            epoch=epoca,
            rectify="full",
            integrate="simpson",
            plot=False,
        )
    if manoejesx.size > 0:
        manoejesx_f2_counts = convert_counts(
            manoejesx_f2,
            time,
            time_scale="ms",
            epoch=epoca,
            rectify="full",
            integrate="simpson",
            plot=False,
        )
        manoejesy_f2_counts = convert_counts(
            manoejesy_f2,
            time,
            time_scale="ms",
            epoch=epoca,
            rectify="full",
            integrate="simpson",
            plot=False,
        )
        manoejesz_f2_counts = convert_counts(
            manoejesz_f2,
            time,
            time_scale="ms",
            epoch=epoca,
            rectify="full",
            integrate="simpson",
            plot=False,
        )
    if piernaejesx.size > 0:
        piernaejesx_f2_counts = convert_counts(
            piernaejesx_f2,
            time,
            time_scale="ms",
            epoch=epoca,
            rectify="full",
            integrate="simpson",
            plot=False,
        )
        piernaejesy_f2_counts = convert_counts(
            piernaejesy_f2,
            time,
            time_scale="ms",
            epoch=epoca,
            rectify="full",
            integrate="simpson",
            plot=False,
        )
        piernaejesz_f2_counts = convert_counts(
            piernaejesz_f2,
            time,
            time_scale="ms",
            epoch=epoca,
            rectify="full",
            integrate="simpson",
            plot=False,
        )
    return (
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
    )
