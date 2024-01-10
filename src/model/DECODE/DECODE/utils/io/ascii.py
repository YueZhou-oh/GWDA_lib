import numpy as np
from returns.result import Failure, Result, Success


def read_data_from_file(file_name: str) -> Result[np.ndarray, str]:
    """
    Reads a matrix from an ASCII file.

    Args:
        file_name (str): The name of the file to read.

    Returns:
        Result[np.ndarray, str]: A matrix of values read from the file, or an error message.
    """
    try:
        with open(file_name) as f:
            data = []
            for line in f:
                row = [float(x) for x in line.split()]
                data.append(row)
        return Success(np.array(data))
    except Exception as e:
        return Failure(str(e))
