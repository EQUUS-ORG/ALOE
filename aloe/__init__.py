import warnings

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    try:
        import torchani
    except:
        warnings.warn("TorchANI is not installed")

    try:
        from .batch_opt.ANI2xt_no_rep import ANI2xt
    except:
        warnings.warn("ANI2xt model is not available")

from .bdfe_calculation.bdfe_calc import get_BDFE, get_G
from .frontend import (
    ConformerConfig,
    OptConfig,
    RankConfig,
    StereoIsoConfig,
    ThermoConfig,
    aloe,
)

__all__ = [
    "aloe",
    "ConformerConfig",
    "OptConfig",
    "RankConfig",
    "StereoIsoConfig",
    "ThermoConfig",
    "get_G",
    "get_BDFE",
]
