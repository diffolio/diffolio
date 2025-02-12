import numpy as np
from typing import Optional


def normalizer(features: np.array, prev_features: np.array, norm_type_: str, dates: Optional[np.array]):
    if norm_type_ == 'ir':  # Increase Ratio (Rate of Increase vs. Previous Date Prices)
        features[1:] = (features[1:] - features[:-1]) / features[:-1]
        features = features[1:]
        np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    elif norm_type_ == 'irvpcp':  # Increase Ratio versus Previous Date Closing Price
        p_c_p = np.expand_dims(features[:-1, :, 3], axis=2)  # Previous Closing Price (T-1, N, 1) except the last step
        features[1:] = (features[1:] - p_c_p) / p_c_p
        features = features[1:]
        np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    elif norm_type_ == 'irvpop':  # Increase Ratio versus Previous Date Opening Price
        p_o_p = np.expand_dims(features[:-1, :, 0], axis=2)  # Previous Opening Price (T-1, N, 1) except the last step
        features[1:] = (features[1:] - p_o_p) / p_o_p
        features = features[1:]
        np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    else:
        raise ValueError(f"Not existing normalization. \'{norm_type_}\'")

    if dates is None:
        pass
    else:
        dates = dates[1:]

    return features, prev_features[1:], dates[1:] if dates is not None else dates
