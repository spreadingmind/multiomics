def transform_df_for_mofa(df, view):
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'sample'}, inplace=True)
    melted_df = df.melt(id_vars='sample', var_name='feature', value_name='value')
    melted_df['view'] = view
    return melted_df
