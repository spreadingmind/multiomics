from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


def transform_df_for_mofa(df, view):
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'sample'}, inplace=True)
    melted_df = df.melt(
        id_vars='sample', var_name='feature', value_name='value')
    melted_df['view'] = view
    return melted_df


def preprocess_data_for_mofa(df):
    # remove all-zero features
    columns_all_zeros = ~df.all(axis=0)
    df = df.loc[:, ~columns_all_zeros]

    # remove features with zero variance
    variances = df.var()
    df = df.loc[:, variances != 0]

    # remove features with largest variance
    threshold = variances.quantile(0.95)
    columns_to_keep = variances[variances <= threshold].index
    df = df[columns_to_keep]

    # normalize
    standard_scaler = StandardScaler()
    df = pd.DataFrame(standard_scaler.fit_transform(
        df), columns=df.columns, index=df.index)

    return df
