from sksurv.preprocessing import encode_categorical  # , OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_percentage_error, classification_report, mean_squared_error
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from IPython.display import display
from src.mofa_utils import transform_df_for_mofa


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
        n_estimators=1000, random_state=RANDOM_STATE, silent=True, cat_features=cat_features_indices)
    cb_classifier_combined.fit(X_train, y_train)

    # Classifier for factors-only dataframe
    cb_classifier_factors = CatBoostClassifier(
        n_estimators=1000, random_state=RANDOM_STATE, silent=True)
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
    plt.figure(figsize=(16, 8))

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


def process_report(report):
    method_metrics = {}
    for key, value in report.items():
        if key in ['macro avg', 'weighted avg']:
            method_metrics[f"{key} f1-score"] = value.get('f1-score', value)
        elif key == 'accuracy':
            pass
        else:
            method_metrics[f"{key} f1-score"] = value['f1-score']
    return method_metrics


def subtype_classification_pipeline_cv(method, X, y, clinical_features, RANDOM_STATE, N_FACTORS, N_NUMERIC_CLINICAL, CLINICAL_FEATURES, plot_feat_imp=True, base_encoder=True, mofa_dataset=None):
    # Разделяем данные на 10 наборов
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    metrics_factors_all, metrics_combined_all = [], []

    if 'mcca' in str(type(method)) or 'autoencoder' in str(method):
        X_stacked = np.hstack([X[0], X[1], X[2]])
        spilts = kf.split(X_stacked)
    else:
        spilts = kf.split(X)

    for train_index, test_index in spilts:
        # Разделяем данные на тренирочную и тестовку подвыборку
        if 'mcca' in str(type(method)) or 'autoencoder' in str(method):
            X_train_k, X_test_k = [x[train_index]
                                   for x in X], [x[test_index] for x in X]
        else:
            X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]
        clinical_train_k, clinical_test_k = clinical_features.iloc[
            train_index], clinical_features.iloc[test_index]

        # Применям метод снижения размерности к мультиомиксным данным
        if method == None:
            X_train_k_reduced = X_train_k
            X_test_k_reduced = X_test_k
        elif 'autoencoder' in str(method):
            if base_encoder == True:
                enc_pipeline = method(
                    X_train_k, N_FACTORS, RANDOM_STATE, base=True)
            else:
                enc_pipeline = method(
                    X_train_k, N_FACTORS, RANDOM_STATE, base=False)
            enc_pipeline.fit()
            X_train_k_reduced = enc_pipeline.transform(X_train_k)
            X_test_k_reduced = enc_pipeline.transform(X_test_k)
        elif 'mofa' in str(method):
            X_train_k_reduced, _, _ = method(
                mofa_dataset, RANDOM_STATE, factors=N_FACTORS)
            X_test_k_reduced = X_train_k_reduced[test_index]
            X_train_k_reduced = X_train_k_reduced[train_index]

        else:
            method.fit(X_train_k)
            X_train_k_reduced = method.transform(X_train_k)
            X_test_k_reduced = method.transform(X_test_k)

        # Модель ргегрессор для факторов и предсказания
        cb_f = CatBoostClassifier(
            n_estimators=1000, random_state=RANDOM_STATE, silent=True)
        cb_f.fit(X_train_k_reduced, y_train_k)
        y_pred_factors_k = cb_f.predict(X_test_k_reduced)
        report_factors = classification_report(
            y_test_k, y_pred_factors_k, output_dict=True)
        metrics_factors_all.append(process_report(report_factors))

        # Соединяем факторы с клиническими данными
        combined_train_k = np.concatenate(
            [X_train_k_reduced, clinical_train_k], axis=1)
        combined_test_k = np.concatenate(
            [X_test_k_reduced, clinical_test_k], axis=1)

        # Модель ргегрессор для факторов с клиническими данными и предсказания
        cat_features_indices = list(
            range(N_FACTORS + N_NUMERIC_CLINICAL, combined_train_k.shape[1]))

        cb_c = CatBoostClassifier(n_estimators=1000, random_state=RANDOM_STATE,
                                  silent=True, cat_features=cat_features_indices)
        cb_c.fit(combined_train_k, y_train_k)
        y_pred_combined_k = cb_c.predict(combined_test_k)
        report_combined = classification_report(
            y_test_k, y_pred_combined_k, output_dict=True)

        metrics_combined_all.append(process_report(report_combined))

    metrics_df = pd.DataFrame({
        'Метрики предсказания по факторам': metrics_factors_all,
        'Метрики предсказания по факторам и клиническим данным': metrics_combined_all
    }, index=[f'Разбиение {k}' for k in range(1, 11)])

    expanded_metrics_df = pd.DataFrame()
    
    for col in metrics_df.columns:
        normalized_df = pd.json_normalize(metrics_df[col])
        normalized_df.columns = pd.MultiIndex.from_product([[col], normalized_df.columns])
        expanded_metrics_df = pd.concat([expanded_metrics_df, normalized_df], axis=1)

    display(expanded_metrics_df)

    metrics = {
        'F1 средневзвешенный предсказания по факторам': np.mean([m['weighted avg f1-score'] for m in metrics_factors_all]),
        'F1 средневзвешенный предсказания по факторам и клиническим данным': np.mean([m['weighted avg f1-score'] for m in metrics_combined_all]),
    }

    if not plot_feat_imp:
        return metrics

    # Отрисовка графика важности признаков
    plt.figure(figsize=(16, 8))

    feature_importances_factors = cb_f.get_feature_importance()
    importances_factors = pd.DataFrame({'Feature': [f'Фактор {i+1}' for i in range(
        len(feature_importances_factors))], 'Importance': feature_importances_factors})
    importances_factors = importances_factors.sort_values(
        by='Importance', ascending=False)

    plt.subplot(2, 2, 4)
    plt.bar(importances_factors['Feature'],
            importances_factors['Importance'], color='blue')
    plt.xticks(rotation=90)
    plt.title('Важность признаков для предсказания подтипа рака по факторам')
    plt.ylabel('Важность')

    feature_importances_combined = cb_c.get_feature_importance()
    feature_names = [
        f'Фактор {i+1}' for i in range(N_FACTORS)] + CLINICAL_FEATURES
    importances_combined = pd.DataFrame(
        {'Feature': feature_names, 'Importance': feature_importances_combined})
    importances_combined = importances_combined.sort_values(
        by='Importance', ascending=False)

    plt.subplot(2, 2, 3)
    plt.bar(importances_combined['Feature'], importances_combined['Importance'], color=np.where(
        importances_combined['Feature'].str.startswith('Фактор'), 'red', 'blue'))
    plt.xticks(rotation=90)
    plt.title(
        'Важность признаков для предсказания подтипа рака по факторам и клиническим данным')
    plt.ylabel('Важность')

    plt.tight_layout()
    plt.show()

    return metrics


def subtype_classification_pipeline(
        factors, clinical_features, cancer_subtype_data_breast, TRAIN_INDICES, TEST_INDICES,
        RANDOM_STATE, N_FACTORS, N_NUMERIC_CLINICAL, CLINICAL_FEATURES, plot_feat_imp=True):
    # Original combined dataframe
    combined_df = pd.concat([pd.DataFrame(factors), clinical_features], axis=1)
    cat_features_indices = list(
        range(N_FACTORS + N_NUMERIC_CLINICAL, combined_df.shape[1]))

    # Factors-only dataframe
    factors_df = pd.DataFrame(factors)

    # Split for combined dataframe
    X_train, X_test, y_train, y_test = combined_df.values[TRAIN_INDICES], combined_df.values[
        TEST_INDICES], cancer_subtype_data_breast.iloc[TRAIN_INDICES], cancer_subtype_data_breast.iloc[TEST_INDICES]

    # Split for factors-only dataframe
    X_train_factors, X_test_factors = factors_df.values[
        TRAIN_INDICES], factors_df.values[TEST_INDICES]

    # Classifier for combined dataframe
    cb_classifier_combined = CatBoostClassifier(
        n_estimators=1000, random_state=RANDOM_STATE, silent=True, cat_features=cat_features_indices)
    cb_classifier_combined.fit(X_train, y_train)

    # Classifier for factors-only dataframe
    cb_classifier_factors = CatBoostClassifier(
        n_estimators=1000, random_state=RANDOM_STATE, silent=True)
    cb_classifier_factors.fit(X_train_factors, y_train)

    # Predictions and evaluations for combined dataframe
    y_pred_combined = cb_classifier_combined.predict(X_test)
    report_combined = classification_report(
        y_test, y_pred_combined, output_dict=True)

    # Predictions and evaluations for factors-only dataframe
    y_pred_factors = cb_classifier_factors.predict(X_test_factors)
    report_factors = classification_report(
        y_test, y_pred_factors, output_dict=True)

    if not plot_feat_imp:
        return report_factors, report_combined

    plt.figure(figsize=(16, 8))

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

    return report_factors, report_combined, feature_importances_factors, feature_importances_combined,


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
        n_estimators=1000, random_state=RANDOM_STATE, silent=True, cat_features=cat_features_indices)
    cb_regressor_combined.fit(X_train, y_train)

    # Regressor for factors-only dataframe
    cb_regressor_factors = CatBoostRegressor(
        n_estimators=1000, random_state=RANDOM_STATE, silent=True)
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
    plt.figure(figsize=(16, 8))

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


def cv_regression_pipeline(method, X, y, clinical_features, death,
                           RANDOM_STATE, N_FACTORS, N_NUMERIC_CLINICAL, CLINICAL_FEATURES, plot_feat_imp=True, base_encoder=True, mofa_dataset=None):
    # Разделяем данные на 10 наборов
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    mape_factors_all, c_index_factors_all = [], []
    mape_combined_all, c_index_combined_all = [], []

    if 'mcca' in str(type(method)) or 'autoencoder' in str(method):
        X_stacked = np.hstack([X[0], X[1], X[2]])
        spilts = kf.split(X_stacked)
    else:
        spilts = kf.split(X)

    for train_index, test_index in spilts:
        # Разделяем данные на тренирочную и тестовку подвыборку
        if 'mcca' in str(type(method)) or 'autoencoder' in str(method):
            X_train_k, X_test_k = [x[train_index]
                                   for x in X], [x[test_index] for x in X]
        else:
            X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]
        clinical_train_k, clinical_test_k = clinical_features.iloc[
            train_index], clinical_features.iloc[test_index]

        # Применям метод снижения размерности к мультиомиксным данным
        if method == None:
            X_train_k_reduced = X_train_k
            X_test_k_reduced = X_test_k
        elif 'autoencoder' in str(method):
            if base_encoder == True:
                enc_pipeline = method(
                    X_train_k, N_FACTORS, RANDOM_STATE, base=True)
            else:
                enc_pipeline = method(
                    X_train_k, N_FACTORS, RANDOM_STATE, base=False)
            enc_pipeline.fit()
            X_train_k_reduced = enc_pipeline.transform(X_train_k)
            X_test_k_reduced = enc_pipeline.transform(X_test_k)
        elif 'mofa' in str(method):
            X_train_k_reduced, _, _ = method(
                mofa_dataset, RANDOM_STATE, factors=N_FACTORS)
            X_test_k_reduced = X_train_k_reduced[test_index]
            X_train_k_reduced = X_train_k_reduced[train_index]

        else:
            method.fit(X_train_k)
            X_train_k_reduced = method.transform(X_train_k)
            X_test_k_reduced = method.transform(X_test_k)

        # Модель ргегрессор для факторов и предсказания
        cb_f = CatBoostRegressor(
            n_estimators=1000, random_state=RANDOM_STATE, silent=True)
        cb_f.fit(X_train_k_reduced, y_train_k)
        y_pred_factors_k = cb_f.predict(X_test_k_reduced)

        # Соединяем факторы с клиническими данными
        combined_train_k = np.concatenate(
            [X_train_k_reduced, clinical_train_k], axis=1)
        combined_test_k = np.concatenate(
            [X_test_k_reduced, clinical_test_k], axis=1)

        # Модель ргегрессор для факторов с клиническими данными и предсказания
        cat_features_indices = list(
            range(N_FACTORS + N_NUMERIC_CLINICAL, combined_train_k.shape[1]))

        cb_c = CatBoostRegressor(n_estimators=1000, random_state=RANDOM_STATE,
                                 silent=True, cat_features=cat_features_indices)
        cb_c.fit(combined_train_k, y_train_k)
        y_pred_combined_k = cb_c.predict(combined_test_k)

        # Считаем метрики качества на
        mape_factors = mean_absolute_percentage_error(
            y_test_k, y_pred_factors_k) * 100
        mape_combined = mean_absolute_percentage_error(
            y_test_k, y_pred_combined_k) * 100
        mape_factors_all.append(mape_factors)
        mape_combined_all.append(mape_combined)

        event = death[test_index].astype(bool)
        c_index_factors = concordance_index_censored(
            event, y_test_k, 1 / y_pred_factors_k)[0]
        c_index_combined = concordance_index_censored(
            event, y_test_k, 1 / y_pred_combined_k)[0]

        c_index_factors_all.append(c_index_factors)
        c_index_combined_all.append(c_index_combined)

    metrics_df = pd.DataFrame({
        'MAPE предсказания по факторам, %': mape_factors_all,
        'MAPE предсказания по факторам и клиническим данным, %': mape_combined_all,
        'C-индекс (цензурированный) предсказания по факторам': c_index_factors_all,
        'C-индекс (цензурированный) предсказания по факторам и клиническим данным': c_index_combined_all,
    }, index=[f'Разбиение {k}' for k in range(1, 11)])

    display(metrics_df)

    metrics = {
        'MAPE_f': np.mean(mape_factors_all),
        'MAPE_f_c': np.mean(mape_combined_all),
        'C-index_f': np.mean(c_index_factors_all),
        'C-index_f_c': np.mean(c_index_combined_all),
    }

    if not plot_feat_imp:
        return metrics

    # Отрисовка графика важности признаков
    plt.figure(figsize=(16, 8))

    feature_importances_factors = cb_f.get_feature_importance()
    importances_factors = pd.DataFrame({'Feature': [f'Фактор {i+1}' for i in range(
        len(feature_importances_factors))], 'Importance': feature_importances_factors})
    importances_factors = importances_factors.sort_values(
        by='Importance', ascending=False)

    plt.subplot(2, 2, 4)
    plt.bar(importances_factors['Feature'],
            importances_factors['Importance'], color='blue')
    plt.xticks(rotation=90)
    plt.title('Важность признаков для предсказания выживаемости по факторам')
    plt.ylabel('Важность')

    feature_importances_combined = cb_c.get_feature_importance()
    feature_names = [
        f'Фактор {i+1}' for i in range(N_FACTORS)] + CLINICAL_FEATURES
    importances_combined = pd.DataFrame(
        {'Feature': feature_names, 'Importance': feature_importances_combined})
    importances_combined = importances_combined.sort_values(
        by='Importance', ascending=False)

    plt.subplot(2, 2, 3)
    plt.bar(importances_combined['Feature'], importances_combined['Importance'], color=np.where(
        importances_combined['Feature'].str.startswith('Фактор'), 'red', 'blue'))
    plt.xticks(rotation=90)
    plt.title(
        'Важность признаков для предсказания выживаемости по факторам и клиническим данным')
    plt.ylabel('Важность')

    plt.tight_layout()
    plt.show()

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
        OneHotEncoder(), CoxPHSurvivalAnalysis(alpha=0.5, n_iter=1000))
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


def cox_regression_pipeline_cv(method, X, y, clinical_features, death,
                               RANDOM_STATE, N_FACTORS, NUMERICAL_CLINICAL_FEATURES, CATEGORICAL_CLINICAL_FEATURES, base_encoder=True, mofa_dataset=None):
    # Делаем дополнительную специальную предобработку клинических данных
    for col in NUMERICAL_CLINICAL_FEATURES:
        clinical_features[col] = clinical_features[col].fillna(-1.)
        clinical_features[col] = clinical_features[col].astype(float)

    # Разделяем данные на 10 наборов
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    c_index_factors_all = []
    c_index_combined_all = []

    if 'mcca' in str(type(method)) or 'autoencoder' in str(method):
        X_stacked = np.hstack([X[0], X[1], X[2]])
        spilts = kf.split(X_stacked)
    else:
        spilts = kf.split(X)

    for train_index, test_index in spilts:
        # Разделяем данные на тренирочную и тестовку подвыборку
        if 'mcca' in str(type(method)) or 'autoencoder' in str(method):
            X_train_k, X_test_k = [x[train_index]
                                   for x in X], [x[test_index] for x in X]
        else:
            X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]
        clinical_train_k, clinical_test_k = clinical_features.iloc[
            train_index], clinical_features.iloc[test_index]
        # Применям метод снижения размерности к мультиомиксным данным
        if method == None:
            X_train_k_reduced = X_train_k
            X_test_k_reduced = X_test_k
        elif 'autoencoder' in str(method):
            if base_encoder == True:
                enc_pipeline = method(
                    X_train_k, N_FACTORS, RANDOM_STATE, base=True)
            else:
                enc_pipeline = method(
                    X_train_k, N_FACTORS, RANDOM_STATE, base=False)
            enc_pipeline.fit()
            X_train_k_reduced = enc_pipeline.transform(X_train_k)
            X_test_k_reduced = enc_pipeline.transform(X_test_k)
        elif 'mofa' in str(method):
            X_train_k_reduced, _, _ = method(
                mofa_dataset, RANDOM_STATE, factors=N_FACTORS)
            X_test_k_reduced = X_train_k_reduced[test_index]
            X_train_k_reduced = X_train_k_reduced[train_index]

        else:
            method.fit(X_train_k)
            X_train_k_reduced = method.transform(X_train_k)
            X_test_k_reduced = method.transform(X_test_k)

        # Преобразуем целевую переменную:
        event_ids_train = death[train_index].astype(bool)
        event_times_train = y_train_k

        event_ids_test = death[test_index].astype(bool)
        event_times_test = y_test_k

        y_train, y_test = list(zip(event_ids_train, event_times_train)), list(
            zip(event_ids_test, event_times_test))

        y_train = np.array(y_train, dtype=[('cens', '?'), ('time', '<f8')])
        y_test = np.array(y_test, dtype=[('cens', '?'), ('time', '<f8')])

        # Модель ргегрессор для факторов и предсказания
        cox_model = CoxPHSurvivalAnalysis(alpha=0.5, n_iter=5000)
        cox_model.fit(X_train_k_reduced, y_train)
        y_pred_factors_k = cox_model.predict(X_test_k_reduced)

        # Соединяем факторы с клиническими данными
        combined_train_k = pd.concat([pd.DataFrame(X_train_k_reduced).set_index(
            clinical_train_k.index), clinical_train_k], axis=1)
        combined_train_k.columns = combined_train_k.columns.astype(str)
        for col in CATEGORICAL_CLINICAL_FEATURES:
            combined_train_k[col] = combined_train_k[col].astype('str')

        ct = ColumnTransformer(
            [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore',
              sparse_output=False), CATEGORICAL_CLINICAL_FEATURES)],
            remainder='passthrough'
        )
        combined_train_k = ct.fit_transform(combined_train_k)

        combined_test_k = pd.concat([pd.DataFrame(X_test_k_reduced).set_index(
            clinical_test_k.index), clinical_test_k], axis=1)
        combined_test_k.columns = combined_test_k.columns.astype(str)
        for col in CATEGORICAL_CLINICAL_FEATURES:
            combined_test_k[col] = combined_test_k[col].astype('str')
        combined_test_k = ct.transform(combined_test_k)

        # Модель ргегрессор для факторов с клиническими данными и предсказания
        cox_model.fit(combined_train_k, y_train)
        y_pred_combined_k = cox_model.predict(combined_test_k)

        # Считаем метрики качества на тестовом наборе
        c_index_factors = concordance_index_censored(
            event_ids_test, event_times_test, y_pred_factors_k)[0]
        # metrics for combined dataframe
        c_index_combined = concordance_index_censored(
            event_ids_test, event_times_test, y_pred_combined_k)[0]

        c_index_factors_all.append(c_index_factors)
        c_index_combined_all.append(c_index_combined)

    metrics_df = pd.DataFrame({
        'C-index (цензурированный) предсказания по факторам': c_index_factors_all,
        'C-index (цензурированный) предсказания по факторам и клиническим данным': c_index_combined_all,
    }, index=[f'Разбиение {k}' for k in range(1, 11)])

    display(metrics_df)

    metrics = {
        'C-index_f': np.mean(c_index_factors_all),
        'C-index_f_c': np.mean(c_index_combined_all),
    }

    return metrics


def regression_pipeline_gdsc(
        factors, target_data, TRAIN_INDICES, TEST_INDICES,
        RANDOM_STATE, plot_feat_imp=True):
    factors_df = pd.DataFrame(factors)

    y_train, y_test = target_data['LN_IC50'][TRAIN_INDICES], target_data['LN_IC50'][TEST_INDICES]

    X_train_factors, X_test_factors = factors_df.values[
        TRAIN_INDICES], factors_df.values[TEST_INDICES]

    # Regressor for factors-only dataframe
    cb_regressor_factors = CatBoostRegressor(
        n_estimators=1000, random_state=RANDOM_STATE, silent=True)
    cb_regressor_factors.fit(X_train_factors, y_train)

    # Predictions and evaluations for factors-only dataframe
    y_pred_factors = cb_regressor_factors.predict(X_test_factors)

    # metrics for factors-only dataframe
    mape_factors = mean_absolute_percentage_error(
        y_true=y_test, y_pred=y_pred_factors) * 100

    rmse_factors = mean_squared_error(
        y_true=y_test, y_pred=y_pred_factors, squared=False)

    metrics = {
        'MAPE_f': mape_factors,
        'RMSE_f': rmse_factors,
    }
    if not plot_feat_imp:
        return metrics
    # Plotting setup
    plt.figure(figsize=(8, 6))

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

    plt.tight_layout()
    plt.show()

    return metrics

def regression_pipeline_gdsc_cv(method, X, y, RANDOM_STATE, N_FACTORS, plot_feat_imp=True, base_encoder=True, mofa_dataset=None):
    # Разделяем данные на 10 наборов
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    mape_factors_all, rmse_factors_all = [], []

    if 'mcca' in str(type(method)) or 'autoencoder' in str(method):
        X_stacked = np.hstack([X[0], X[1], X[2]])
        spilts = kf.split(X_stacked)
    else:
        spilts = kf.split(X)

    for train_index, test_index in spilts:
        # Разделяем данные на тренирочную и тестовку подвыборку
        if 'mcca' in str(type(method)) or 'autoencoder' in str(method):
            X_train_k, X_test_k = [x[train_index]
                                   for x in X], [x[test_index] for x in X]
        else:
            X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]

        # Применям метод снижения размерности к мультиомиксным данным
        if method == None:
            X_train_k_reduced = X_train_k
            X_test_k_reduced = X_test_k
        elif 'autoencoder' in str(method):
            if base_encoder == True:
                enc_pipeline = method(
                    X_train_k, N_FACTORS, RANDOM_STATE, base=True)
            else:
                enc_pipeline = method(
                    X_train_k, N_FACTORS, RANDOM_STATE, base=False)
            enc_pipeline.fit()
            X_train_k_reduced = enc_pipeline.transform(X_train_k)
            X_test_k_reduced = enc_pipeline.transform(X_test_k)
        elif 'mofa' in str(method):
            X_train_k_reduced, _, _ = method(
                mofa_dataset, RANDOM_STATE, factors=N_FACTORS)
            X_test_k_reduced = X_train_k_reduced[test_index]
            X_train_k_reduced = X_train_k_reduced[train_index]

        else:
            method.fit(X_train_k)
            X_train_k_reduced = method.transform(X_train_k)
            X_test_k_reduced = method.transform(X_test_k)

        # Модель ргегрессор для факторов и предсказания
        cb_f = CatBoostRegressor(
            n_estimators=1000, random_state=RANDOM_STATE, silent=True)
        cb_f.fit(X_train_k_reduced, y_train_k)
        y_pred_factors_k = cb_f.predict(X_test_k_reduced)

        # Считаем метрики качества на
        mape_factors = mean_absolute_percentage_error(
            y_test_k, y_pred_factors_k) * 100
        rmse_factors = mean_squared_error(
            y_test_k, y_pred=y_pred_factors_k, squared=False)
        mape_factors_all.append(mape_factors)
        rmse_factors_all.append(rmse_factors)

    metrics_df = pd.DataFrame({
        'MAPE предсказания по факторам, %': mape_factors_all,
        'MAPE предсказания по факторам': rmse_factors_all,
    }, index=[f'Разбиение {k}' for k in range(1, 11)])

    display(metrics_df)

    metrics = {
        'MAPE_f': np.mean(mape_factors_all),
        'RMSE_f': np.mean(rmse_factors_all),
    }
    if not plot_feat_imp:
        return metrics

    # Отрисовка графика важности признаков
    plt.figure(figsize=(16, 8))

    feature_importances_factors = cb_f.get_feature_importance()
    importances_factors = pd.DataFrame({'Feature': [f'Фактор {i+1}' for i in range(
        len(feature_importances_factors))], 'Importance': feature_importances_factors})
    importances_factors = importances_factors.sort_values(
        by='Importance', ascending=False)

    plt.subplot(2, 2, 4)
    plt.bar(importances_factors['Feature'],
            importances_factors['Importance'], color='blue')
    plt.xticks(rotation=90)
    plt.title('Важность признаков для предсказания выживаемости по факторам')
    plt.ylabel('Важность')

    plt.tight_layout()
    plt.show()

    return metrics
