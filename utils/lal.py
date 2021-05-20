
from lal import MSUN_SI
from lalsimulation import (
    SimInspiralTransformPrecessingNewInitialConditions,
    GetApproximantFromString,
    SimInspiralImplementedFDApproximants
)

def source_frame_to_radiation(
    theta_jn: float, phi_jl: float, tilt_1: float, tilt_2: float, phi_12: float,
    a_1: float, a_2: float, mass_1: float, mass_2: float, f_ref: float, phase: float
):
    # convert masses from Mpc to SI units
    mass_1_SI = mass_1 * MSUN_SI
    mass_2_SI = mass_2 * MSUN_SI

    # Following bilby code
    if (
        (a_1 == 0.0 or tilt_1 in [0, np.pi])
        and (a_2 == 0.0 or tilt_2 in [0, np.pi])
    ):
        spin_1x, spin_1y, spin_1z = 0.0, 0.0, a_1 * np.cos(tilt_1)
        spin_2x, spin_2y, spin_2z, = 0.0, 0.0, a_2 * np.cos(tilt_2)
        iota = theta_jn
    else:
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (
            SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                a_1, a_2, mass_1_SI, mass_2_SI, f_ref, phase
            )
        )
    return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z

def is_fd_waveform(approximant: str) -> bool:
    """Return whether the approximant is implemented in the frequency domain (FD).

    Args:
        approximant (str): name of approximant according to LALSimulation
    """
    # LAL refers to approximants by an index
    lal_num = GetApproximantFromString(approximant)
    return bool(SimInspiralImplementedFDApproximants(lal_num))