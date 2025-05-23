import numpy as np

def json_encode_np(obj):
    """
    JSON can't serialize numpy types, so convert to pure python
    """
    if isinstance(obj, np.ndarray):
        return list(obj)
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, int32):
        return int(obj)
    elif isinstance(obj, int64):
        return int(obj)
    else:
        return obj
