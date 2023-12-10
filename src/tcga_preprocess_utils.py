import pandas as pd


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
