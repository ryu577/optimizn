import numpy as np
import pandas as pd
import time
from datetime import datetime
from optimizn.ab_split.testing.cluster_vmsku import cluster_vmsku
from optimizn.ab_split.opt_split import form_arrays


def main1():
    arrays, vm_ix, cl_ix =\
            form_arrays(cluster_vmsku, "Usage_VMSize")
    n_ros = len(arrays)
    n_cols = len(arrays[0])
    index = [i for i in range(n_ros * n_cols)]
    m_df = pd.DataFrame(columns=['matDat', 'roIx', 'coIx', 'matId'],
                        index=index)
    m_id = int(time.time())
    ixx = 0
    for i1 in range(len(arrays)):
        arr = arrays[i1]
        for i2 in range(len(arr)):
            dd = arr[i2]
            m_df.loc[ixx] = [dd, i1, i2, m_id]
            ixx += 1
    index = [i for i in range(len(vm_ix) + len(cl_ix))]
    map_df = pd.DataFrame(columns=['matId', 'IxType', 'Ix', 'Name', 'Value'],
                          index=index)
    ix = 0
    for kk in vm_ix.keys():
        map_df.loc[ix] = [m_id, 'row', vm_ix[kk], kk]
        ix += 1
    for kk in cl_ix.keys():
        map_df.loc[ix] = [m_id, 'col', cl_ix[kk], kk]
        ix += 1
    t1 = datetime.now()
    day = t1.day
    month = t1.month
    year = t1.year
    map_df.to_csv('Data/OptSplit/Reader/PropertyIx/' + str(year) + str(month) +
                  str(day) + '_' + str(m_id) + '.csv', index=False)
    m_df.to_csv('Data/OptSplit/Reader/Matrix/' + str(year) + str(month) +
                str(day) + '_' + str(m_id) + '.csv', index=False)
