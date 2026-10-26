"""APIMS ten-feature preprocessing; inputs are ordered per the monograph."""
import numpy as np

FEATURES = ('cpu_temp_c','pcie_correctable','pcie_uncorrectable','ecc_single','ecc_multi','nvme_wear_pct','nvme_reallocated','fan_rpm','rail_12v','optical_dbm')

def normalize(raw):
    """Normalize arrays with trailing dimension 10; no clipping is applied."""
    x = np.asarray(raw, dtype=np.float32)
    if x.shape[-1] != 10 or not np.isfinite(x).all():
        raise ValueError('Expected finite telemetry with final dimension 10')
    if np.any(x[..., [1,2,3,4,5,6]] < 0):
        raise ValueError('Counts and wear percentage must be nonnegative')
    z = np.empty_like(x)
    z[...,0] = (x[...,0]-45)/40
    z[...,1] = np.log1p(x[...,1])
    z[...,2] = (x[...,2]>0).astype(np.float32)
    z[...,3] = np.log1p(x[...,3])
    z[...,4] = (x[...,4]>0).astype(np.float32)
    z[...,5] = x[...,5]/100
    z[...,6] = np.log1p(x[...,6])
    z[...,7] = (x[...,7]-4000)/6000
    z[...,8] = (x[...,8]-12)/0.6
    z[...,9] = (x[...,9]+10)/10
    return z

def windows(series, length=128):
    """Create consecutive windows; resample/align irregular telemetry upstream."""
    x = np.asarray(series, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 10 or len(x)<length:
        raise ValueError('Expected (time, 10) with at least length observations')
    return np.stack([x[i:i+length] for i in range(len(x)-length+1)])
