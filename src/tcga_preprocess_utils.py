import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_STATE = 42

def prepare_survival_data(path, id_sep='.'):
    survival_data = pd.read_csv(path, sep='\t')
    survival_data['PatientID'] = survival_data['PatientID'].str.lower()
    
    if id_sep == '-':
        survival_data['PatientID'] = survival_data['PatientID'].apply(lambda x: "-".join(x.split("-")[:-1]))
        survival_data['PatientID'] = survival_data['PatientID'].str.replace('-', '.')

    survival_data = handle_duplicates(survival_data)
    survival_data = survival_data.dropna(subset=['Death'])

    survival_data = survival_data.set_index('PatientID')
    survival_data = survival_data.sort_index()
    return survival_data

def prepare_clinical_data(path):
    clinical_data = pd.read_csv(path, sep='\t')
    clinical_data['sampleID'] = clinical_data['sampleID'].str.lower()
    
    clinical_data['sampleID'] = clinical_data['sampleID'].apply(lambda x: "-".join(x.split("-")[:-1]))
    clinical_data['sampleID'] = clinical_data['sampleID'].str.replace('-', '.')

    clinical_data = clinical_data.drop_duplicates(subset='sampleID')

    clinical_data = clinical_data.set_index('sampleID')
    clinical_data = clinical_data.sort_index()

    return clinical_data


def handle_duplicates(df):
    inconsistent_patient_ids = []

    for patient_id, group in df.groupby('PatientID'):
        if group['Death'].nunique() > 1:
            inconsistent_patient_ids.append(patient_id)

    if inconsistent_patient_ids:
        print("Inconsistent Patient IDs:", inconsistent_patient_ids)
        df = df[~df['PatientID'].isin(inconsistent_patient_ids)]
    else:
        df = df.drop_duplicates(subset='PatientID')

    return df


def prepare_data(path, patient_ids=None):
    data = pd.read_csv(path, sep=' ')

    data = data.T
    data.index = data.index.str.lower()
    data.index = data.index.str.replace(r'\.01$', '', regex=True)

    if patient_ids is not None:
        data = data[data.index.isin(patient_ids)]
    data = data.sort_index()
    return data


def choose_common_patients(dfs):
    if len(dfs) == 4:
        df1, df2, df3, df4 = dfs
        common_indexes = set(df1.index) & set(
            df2.index) & set(df3.index) & set(df4.index)

        df1_sync = df1[df1.index.isin(common_indexes)]
        df2_sync = df2[df2.index.isin(common_indexes)]
        df3_sync = df3[df3.index.isin(common_indexes)]
        df4_sync = df4[df4.index.isin(common_indexes)]

        return df1_sync, df2_sync, df3_sync, df4_sync
    elif len(dfs) == 3:
        df1, df2, df3 = dfs
        common_indexes = set(df1.index) & set(
            df2.index) & set(df3.index)

        df1_sync = df1[df1.index.isin(common_indexes)]
        df2_sync = df2[df2.index.isin(common_indexes)]
        df3_sync = df3[df3.index.isin(common_indexes)]

        return df1_sync, df2_sync, df3_sync
    elif len(dfs) == 6:
        df1, df2, df3, df4, df5, df6 = dfs
        common_indexes = set(df1.index) & set(
            df2.index) & set(df3.index) & set(
            df4.index) & set(df5.index) & set(
            df6.index)

        df1_sync = df1[df1.index.isin(common_indexes)]
        df2_sync = df2[df2.index.isin(common_indexes)]
        df3_sync = df3[df3.index.isin(common_indexes)]
        df4_sync = df4[df4.index.isin(common_indexes)]
        df5_sync = df5[df5.index.isin(common_indexes)]
        df6_sync = df6[df6.index.isin(common_indexes)]

        return df1_sync, df2_sync, df3_sync, df4_sync, df5_sync, df6_sync


def split_patients(n_breast=685, n_kidney=208, test_size=0.25, random_state=RANDOM_STATE):
    target_cancer_type = [0] * n_breast
    target_kidney = [1] * n_kidney
    n_samples = n_breast + n_kidney

    target_cancer_type.extend(target_kidney)

    indices = range(n_samples)
    indices_train, indices_test = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=target_cancer_type)

    return indices_train, indices_test, np.array(target_cancer_type)

def split_patients_for_target_prediction(target_df, test_size=0.25, random_state=RANDOM_STATE, stratify_by='Death'):
    stratify = None
    if stratify_by:
        stratify = target_df[stratify_by]
    indices_train, indices_test = train_test_split(
        target_df.index, test_size=test_size, random_state=random_state, stratify=stratify)

    return indices_train, indices_test