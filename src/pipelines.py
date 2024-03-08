from sksurv.preprocessing import OneHotEncoder, encode_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_percentage_error
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.pipeline import make_pipeline
from sksurv.preprocessing import OneHotEncoder


def classification_pipeline(
        factors, clinical_features, survival_data_breast, TRAIN_INDICES, TEST_INDICES,
        RANDOM_STATE, N_FACTORS, N_NUMERIC_CLINICAL, CLINICAL_FEATURES, plot_feat_imp=True):
    # Original combined dataframe
    combined_df = pd.concat([pd.DataFrame(factors), clinical_features], axis=1)
    cat_features_indices = list(
        range(N_FACTORS + N_NUMERIC_CLINICAL, combined_df.shape[1]))

    # Factors-only dataframe
    factors_df = pd.DataFrame(factors)

    # Split for combined dataframe
    X_train, X_test, y_train, y_test = combined_df.values[TRAIN_INDICES], combined_df.values[TEST_INDICES], survival_data_breast[
        'Death'][TRAIN_INDICES], survival_data_breast['Death'][TEST_INDICES]

    # Split for factors-only dataframe
    X_train_factors, X_test_factors = factors_df.values[
        TRAIN_INDICES], factors_df.values[TEST_INDICES]

    # Classifier for combined dataframe
    cb_classifier_combined = CatBoostClassifier(
        n_estimators=5000, random_state=RANDOM_STATE, silent=True, cat_features=cat_features_indices)
    cb_classifier_combined.fit(X_train, y_train)

    # Classifier for factors-only dataframe
    cb_classifier_factors = CatBoostClassifier(
        n_estimators=5000, random_state=RANDOM_STATE, silent=True)
    cb_classifier_factors.fit(X_train_factors, y_train)

    # Predictions and evaluations for combined dataframe
    y_pred_combined = cb_classifier_combined.predict(X_test)
    y_probas_combined = cb_classifier_combined.predict_proba(X_test)
    roc_auc_combined = roc_auc_score(y_test, y_probas_combined[:, 1])

    # Predictions and evaluations for factors-only dataframe
    y_pred_factors = cb_classifier_factors.predict(X_test_factors)
    y_probas_factors = cb_classifier_factors.predict_proba(X_test_factors)
    roc_auc_factors = roc_auc_score(y_test, y_probas_factors[:, 1])

    # Plotting setup
    plt.figure(figsize=(20, 12))

    # ROC-AUC for factors-only dataframe
    fpr_factors, tpr_factors, thresholds_factors = roc_curve(
        y_test.values, y_probas_factors[:, 1])
    plt.subplot(2, 2, 2)
    plt.plot(fpr_factors, tpr_factors)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'Factors ROC-AUC: {np.round(roc_auc_factors, 3)}')
    plt.fill_between(fpr_factors, tpr_factors, color="r", alpha=0.3)

    # ROC-AUC for combined dataframe
    fpr_combined, tpr_combined, thresholds_combined = roc_curve(
        y_test.values, y_probas_combined[:, 1])
    plt.subplot(2, 2, 1)
    plt.plot(fpr_combined, tpr_combined)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'Combined ROC-AUC: {np.round(roc_auc_combined, 3)}')
    plt.fill_between(fpr_combined, tpr_combined, color="b", alpha=0.3)

    if not plot_feat_imp:
        return roc_auc_factors, roc_auc_combined

    # Feature importances for factors-only dataframe
    feature_importances_factors = cb_classifier_factors.get_feature_importance()
    importances_factors = pd.DataFrame({'Feature': [f'Factor {i+1}' for i in range(
        len(feature_importances_factors))], 'Importance': feature_importances_factors})
    importances_factors = importances_factors.sort_values(
        by='Importance', ascending=False)

    plt.subplot(2, 2, 4)
    plt.bar(importances_factors['Feature'],
            importances_factors['Importance'], color='blue')
    plt.xticks(rotation=90)
    plt.title('Feature Importances Factors Only')
    plt.ylabel('Importance')

    # Feature importances for combined dataframe
    feature_importances_combined = cb_classifier_combined.get_feature_importance()
    feature_names = [
        f'Factor {i+1}' for i in range(N_FACTORS)] + CLINICAL_FEATURES
    importances_combined = pd.DataFrame(
        {'Feature': feature_names, 'Importance': feature_importances_combined})
    importances_combined = importances_combined.sort_values(
        by='Importance', ascending=False)

    plt.subplot(2, 2, 3)
    plt.bar(importances_combined['Feature'], importances_combined['Importance'], color=np.where(
        importances_combined['Feature'].str.startswith('Factor'), 'red', 'blue'))
    plt.xticks(rotation=90)
    plt.title('Feature Importances Combined')
    plt.ylabel('Importance')

    plt.tight_layout()
    plt.show()

    return roc_auc_factors, roc_auc_combined, feature_importances_factors, feature_importances_combined,


def regression_pipeline(
        factors, clinical_features, survival_data_breast, TRAIN_INDICES, TEST_INDICES,
        RANDOM_STATE, N_FACTORS, N_NUMERIC_CLINICAL, CLINICAL_FEATURES, plot_feat_imp=True):
    # Original combined dataframe
    combined_df = pd.concat([pd.DataFrame(factors), clinical_features], axis=1)
    cat_features_indices = list(
        range(N_FACTORS + N_NUMERIC_CLINICAL, combined_df.shape[1]))

    # Factors-only dataframe
    factors_df = pd.DataFrame(factors)

    # Split for combined dataframe
    X_train, X_test, y_train, y_test = combined_df.values[TRAIN_INDICES], combined_df.values[TEST_INDICES], survival_data_breast[
        'Survival'][TRAIN_INDICES], survival_data_breast['Survival'][TEST_INDICES]

    # Split for factors-only dataframe
    X_train_factors, X_test_factors = factors_df.values[
        TRAIN_INDICES], factors_df.values[TEST_INDICES]

    # Regressor for combined dataframe
    cb_regressor_combined = CatBoostRegressor(
        n_estimators=5000, random_state=RANDOM_STATE, silent=True, cat_features=cat_features_indices)
    cb_regressor_combined.fit(X_train, y_train)

    # Regressor for factors-only dataframe
    cb_regressor_factors = CatBoostRegressor(
        n_estimators=5000, random_state=RANDOM_STATE, silent=True)
    cb_regressor_factors.fit(X_train_factors, y_train)

    # Predictions and evaluations for combined dataframe
    y_pred_combined = cb_regressor_combined.predict(X_test)
    # Predictions and evaluations for factors-only dataframe
    y_pred_factors = cb_regressor_factors.predict(X_test_factors)

    event = survival_data_breast['Death'][TEST_INDICES].astype(bool)
    # metrics for factors-only dataframe
    mape_factors = mean_absolute_percentage_error(
        y_true=y_test, y_pred=y_pred_factors) * 100
    c_index_factors = concordance_index_censored(
        event, y_test, 1 / y_pred_factors)[0]

    # metrics for combined dataframe
    mape_combined = mean_absolute_percentage_error(
        y_true=y_test, y_pred=y_pred_combined) * 100
    c_index_combined = concordance_index_censored(
        event, y_test, 1 / y_pred_combined)[0]

    metrics = {
        'MAPE_f': mape_factors,
        'MAPE_f_c': mape_combined,
        'C-index_f': c_index_factors,
        'C-index_f_c': c_index_combined,
    }
    if not plot_feat_imp:
        return metrics
    # Plotting setup
    plt.figure(figsize=(20, 12))

    # Feature importances for factors-only dataframe
    feature_importances_factors = cb_regressor_factors.get_feature_importance()
    importances_factors = pd.DataFrame({'Feature': [f'Factor {i+1}' for i in range(
        len(feature_importances_factors))], 'Importance': feature_importances_factors})
    importances_factors = importances_factors.sort_values(
        by='Importance', ascending=False)

    plt.subplot(2, 2, 4)
    plt.bar(importances_factors['Feature'],
            importances_factors['Importance'], color='blue')
    plt.xticks(rotation=90)
    plt.title('Feature Importances Factors Only')
    plt.ylabel('Importance')

    # Feature importances for combined dataframe
    feature_importances_combined = cb_regressor_combined.get_feature_importance()
    feature_names = [
        f'Factor {i+1}' for i in range(N_FACTORS)] + CLINICAL_FEATURES
    importances_combined = pd.DataFrame(
        {'Feature': feature_names, 'Importance': feature_importances_combined})
    importances_combined = importances_combined.sort_values(
        by='Importance', ascending=False)

    plt.subplot(2, 2, 3)
    plt.bar(importances_combined['Feature'], importances_combined['Importance'], color=np.where(
        importances_combined['Feature'].str.startswith('Factor'), 'red', 'blue'))
    plt.xticks(rotation=90)
    plt.title('Feature Importances Combined')
    plt.ylabel('Importance')

    plt.tight_layout()
    plt.show()

    return metrics


def cox_regression_pipeline(
        factors, clinical_features, survival_data_breast, TRAIN_INDICES, TEST_INDICES,
        RANDOM_STATE, N_FACTORS, N_NUMERIC_CLINICAL, CLINICAL_FEATURES, plot_feat_imp=False):
    # Original combined dataframe
    combined_df = pd.concat([pd.DataFrame(factors), clinical_features], axis=1)
    cat_features_indices = list(
        range(N_FACTORS + N_NUMERIC_CLINICAL, combined_df.shape[1]))

    # Factors-only dataframe
    factors_df = pd.DataFrame(factors)

    # Split for combined dataframe
    X_train, X_test, y_train, y_test = combined_df.loc[[TRAIN_INDICES]], combined_df.loc[[TEST_INDICES]], survival_data_breast[
        'Survival'][TRAIN_INDICES], survival_data_breast['Survival'][TEST_INDICES]

    # Split for factors-only dataframe
    X_train_factors, X_test_factors = factors_df.loc[
        [TRAIN_INDICES]], factors_df.loc[[TEST_INDICES]]

    # Model for combined dataframe
    cox_model_combined = make_pipeline(
        OneHotEncoder(), CoxPHSurvivalAnalysis())
    cox_model_combined.fit(X_train, y_train)

    # Model for factors-only dataframe
    cox_model_factors = CoxPHSurvivalAnalysis()
    cox_model_factors.fit(X_train_factors, y_train)

    # Predictions and evaluations for combined dataframe
    y_pred_combined = cox_model_combined.predict(X_test)
    # Predictions and evaluations for factors-only dataframe
    y_pred_factors = cox_model_factors.predict(X_test_factors)

    # metrics for factors-only dataframe
    c_index_factors = concordance_index_censored(
        survival_data_breast['Death'][TEST_INDICES].astype(bool), y_test, y_pred_factors)[0]

    # metrics for combined dataframe
    c_index_combined = concordance_index_censored(
        survival_data_breast['Death'][TEST_INDICES].astype(bool), y_test, y_pred_combined)[0]

    metrics = {
        'C-index_f': c_index_factors,
        'C-index_f_c': c_index_combined,
    }
    return metrics


def cox_regression_pipeline(
    factors, clinical_features, survival_data_breast, TRAIN_INDICES, TEST_INDICES,
        NUMERICAL_CLINICAL_FEATURES, CATEGORICAL_CLINICAL_FEATURES):

    clinical_features_ = clinical_features.copy()
    for col in NUMERICAL_CLINICAL_FEATURES:
        clinical_features_[col] = clinical_features_[col].fillna(-1.)
        clinical_features_[col] = clinical_features_[col].astype(float)

    # Original combined dataframe
    combined_df = pd.concat(
        [pd.DataFrame(factors), clinical_features_], axis=1)
    combined_df.columns = combined_df.columns.astype(str)

    for col in CATEGORICAL_CLINICAL_FEATURES:
        combined_df[col] = combined_df[col].astype('category')

    # Factors-only dataframe
    factors_df = pd.DataFrame(factors)
    factors_df.columns = factors_df.columns.astype(str)

    # Split for combined dataframe
    event_ids_train = survival_data_breast['Death'][TRAIN_INDICES].astype(
        bool).values
    event_times_train = survival_data_breast['Survival'][TRAIN_INDICES].values

    event_ids_test = survival_data_breast['Death'][TEST_INDICES].astype(
        bool).values
    event_times_test = survival_data_breast['Survival'][TEST_INDICES].values

    X_train, X_test, y_train, y_test = combined_df.loc[TRAIN_INDICES], combined_df.loc[TEST_INDICES], \
        list(zip(event_ids_train, event_times_train)), \
        list(zip(event_ids_test, event_times_test))

    y_train = np.array(y_train, dtype=[('cens', '?'), ('time', '<f8')])
    y_test = np.array(y_test, dtype=[('cens', '?'), ('time', '<f8')])

    # Split for factors-only dataframe
    X_train_factors, X_test_factors = factors_df.loc[
        TRAIN_INDICES], factors_df.loc[TEST_INDICES]
    # Model for combined dataframe
    cox_model_combined = make_pipeline(
        OneHotEncoder(), CoxPHSurvivalAnalysis(alpha=0.5, n_iter=5000))
    cox_model_combined.fit(X_train, y_train)

    # Model for factors-only dataframe
    cox_model_factors = CoxPHSurvivalAnalysis(alpha=0.5, n_iter=5000)
    cox_model_factors.fit(X_train_factors, y_train)

    # Predictions and evaluations for combined dataframe
    y_pred_combined = cox_model_combined.predict(X_test)
    # Predictions and evaluations for factors-only dataframe
    risk_pred_factors = cox_model_factors.predict(X_test_factors)

    # metrics for factors-only dataframe
    c_index_factors = concordance_index_censored(
        event_ids_test, event_times_test, risk_pred_factors)[0]
    # metrics for combined dataframe
    c_index_combined = concordance_index_censored(
        event_ids_test, event_times_test, y_pred_combined)[0]

    metrics = {
        'C-index_f': c_index_factors,
        'C-index_f_c': c_index_combined,
    }
    return metrics