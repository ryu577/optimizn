import numpy as np
import pandas as pd
import os
from optimizn.ab_split.opt_split_heuristic.splitter import df_to_mat


def get_mat(m_id):
    files = os.listdir('Data/OptSplit/Reader/Matrix/')
    file1 = sorted(files)[len(files)-1]
    file2 = sorted(files)[len(files)-2]
    file3 = sorted(files)[len(files)-3]
    mat_df1 = pd.read_csv('Data/OptSplit/Reader/Matrix/' + file1)
    mat_df2 = pd.read_csv('Data/OptSplit/Reader/Matrix/' + file2)
    mat_df3 = pd.read_csv('Data/OptSplit/Reader/Matrix/' + file3)
    mat_df1 = mat_df1[mat_df1.matId == m_id]
    mat_df2 = mat_df2[mat_df2.matId == m_id]
    mat_df3 = mat_df3[mat_df3.matId == m_id]
    for mat_df in [mat_df1, mat_df2, mat_df3]:
        if len(mat_df) != 0:
            break
    return mat_df


files = os.listdir('Data/OptSplit/Splitter/Permutation/')
file1 = sorted(files)[len(files)-1]
permut_df = pd.read_csv('Data/OptSplit/Splitter/Permutation/' + file1)
m_id = permut_df.matId[0]

files = os.listdir('Data/OptSplit/Splitter/Split/')
file1 = sorted(files)[len(files)-1]
splt_df = pd.read_csv('Data/OptSplit/Splitter/Split/' + file1)

mat_df = get_mat(m_id)
mat1 = df_to_mat(mat_df)

# Now read the row permutations and permute mat1.
ro_perm = permut_df[permut_df.IxType == "row"]
perm_arr = ro_perm.Ix

# Then read the column permutations and permute mat1.
