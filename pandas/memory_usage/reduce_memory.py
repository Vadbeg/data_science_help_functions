"""
Module for memory reduction in pandas dataframes

@by Vadbeg[07.02.2020]
"""


import time

import pandas as pd
import numpy as np


def reduce_mem_usage(df, verbose=True):
    """
    Reduces memory usage with help of
    variable type downsampling

    :param df: pandas dataframe
    :type df: pd.DataFrame
    :param verbose: if True -> prints info about memory usage
    :type verbose: bool
    :return: processed dataframe
    :rtype: pd.DataFrame
    """

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2
    start_time = time.time()

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    end_time = time.time()

    if verbose:
        memory_reduction_percent = 100 * (start_mem - end_mem) / start_mem
        time_usage = end_time - start_time

        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
                                                                              memory_reduction_percent))
        print(f'Time usage: {time_usage * 1000:.4f} milliseconds')

    return df


if __name__ == '__main__':
    ints64 = np.random.randint(0, 3000, size=(4, 5), dtype=np.int64)
    floats64 = np.random.randn(4, 5).astype('float64')

    df_int = pd.DataFrame(data=ints64)
    df_float = pd.DataFrame(data=floats64)

    df = df_int.join(df_float, lsuffix='_')

    df_res = reduce_mem_usage(df, verbose=True)

    print(df.dtypes)
    print(df_res.dtypes)
