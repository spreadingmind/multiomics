from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from mofapy2.run.entry_point import entry_point
import h5py
import time


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


def train_mofa(data_df, random_state=42, factors=10, train_params={'iter': 5000, 'convergence_mode': 'slow'}):
    ent = entry_point()
    # (2) Set data options
    # - scale_views: if views have very different ranges, one can to scale each view to unit variance
    ent.set_data_options(
        scale_views=False
    )

    # (3) Set data using the data frame format
    ent.set_data_df(data_df)

    # using personalised values
    ent.set_model_options(
        factors=factors,
        ard_factors=True
    )

    ## (5) Set training options ##
    # - iter: number of iterations
    # - convergence_mode: "fast", "medium", "slow". Fast mode is usually good enough.
    # - dropR2: minimum variance explained criteria to drop factors while training. Default is None, inactive factors are not dropped during training
    # - gpu_mode: use GPU mode? this functionality needs cupy installed and a functional GPU, see https://biofam.github.io/MOFA2/gpu_training.html
    # - seed: random seed

    # using default values
    ent.set_train_options()

    # using personalised values
    ent.set_train_options(
        **train_params,
        gpu_mode=False,
        seed=random_state
    )

    ####################################
    ## Build and train the MOFA model ##
    ####################################

    # Build the model
    ent.build()

    # Run the model
    ent.run()

    ####################
    ## Save the model ##
    ####################

    outfile = f"data/outputs/test_{time.time()}.hdf5"

    # - save_data: logical indicating whether to save the training data in the hdf5 file.
    # this is useful for some downstream analysis in R, but it can take a lot of disk space.
    ent.save(outfile, save_data=True)

    #########################
    ## Downstream analysis ##
    #########################

    # Check the mofax package for the downstream analysis in Python: https://github.com/bioFAM/mofax
    # Check the MOFA2 R package for the downstream analysis in R: https://www.bioconductor.org/packages/release/bioc/html/MOFA2.html
    # All tutorials: https://biofam.github.io/MOFA2/tutorials.html

    # Extract factor values (a list with one matrix per sample group)
    factors = ent.model.nodes["Z"].getExpectation()

    # Extract weights  (a list with one matrix per view)
    weights = ent.model.nodes["W"].getExpectation()

    # Extract variance explained values
    r2 = ent.model.calculate_variance_explained()

    return factors, weights, r2
