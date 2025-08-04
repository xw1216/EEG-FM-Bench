import json
import pickle
from typing import Union

import h5py
import numpy as np


class DataStorage:
    """Handler for saving/loading dictionaries with mixed types (ndarray, str, int)"""

    @staticmethod
    def save_h5(data: dict[str, Union[np.ndarray, str, int]], filename: str) -> None:
        with h5py.File(filename, 'w') as f:
            # Save metadata about types
            dtypes = {k: str(type(v).__name__) for k, v in data.items()}
            f.attrs['dtypes'] = json.dumps(dtypes)

            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(f"data/{key}", data=value, compression="gzip", compression_opts=9)
                else:
                    if isinstance(value, str):
                        f.attrs[key] = value
                    else:
                        f.attrs[key] = value

    @staticmethod
    def load_h5(filename: str) -> dict[str, Union[np.ndarray, str, int]]:
        """Load data from HDF5"""
        result = {}
        with h5py.File(filename, 'r') as f:
            # Load scalar values from attributes
            for key in f.attrs.keys():
                if key != 'dtypes':
                    result[key] = f.attrs[key]

            # Load arrays from datasets
            if 'data' in f:
                for key in f['data'].keys():
                    result[key] = f['data'][key][()]

        return result

    @staticmethod
    def save_pickle(data: dict[str, Union[np.ndarray, str, int]], filename: str) -> None:
        """Save using pickle - simpler but loads entire data at once"""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_pickle(filename: str) -> dict[str, Union[np.ndarray, str, int]]:
        """Load data from pickle"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
