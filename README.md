## Project info
1. `tcga_preprocess.ipynb`

Preprocessing of 2 TCGA datasets: breast and kidney. In the result we get data for downstream analysis for dimemtionality reduction algorithms.

2. `tcga_breast.ipynb`

Deriving factors from statistical models and building cancer survival regressor.



## Dimentionality Reduction Algorithms info

## Relevant papers and tools

- [x] Benchmarking joint multi-omics dimensionality reduction approaches for the study of cancer: https://www.nature.com/articles/s41467-020-20430-7
    - Наша первая исходная ознакомительная статья
    - Ссылка на датасет TCGA

- [x] BENCHMARKING OF JOINT DIMENSIONALITY REDUCTION METHODS FOR THE ANALYSIS OF MULTI-OMICS DATA IN CANCER: https://libstore.ugent.be/fulltxt/RUG01/003/008/173/RUG01-003008173_2021_0001_AC.pdf
    - 2nd dataset: The Cancer Cell Line Encyclopedia (CCLE) dataset: https://sites.broadinstitute.org/ccle/datasets
    - TCGA clinical data provides survival time in days from the initial pathological diagnosis until the death of a patient (TCGA metadata, n.d.). Survival data is stored as “days
    - Делали survival analysis по факторам от jDR (см секцию 3.2 Association of jDR factors with survival, page 31):
    Cox proportional hazards regression model was used to testing the association between survival and jDR factors because the model allows us to test the association between several risk factors and survival. “Coxph” function provided by the “survival” package in R was used to fit the models. P-values indicating the association between the factors to survival were adjusted for multiple testing by using the Bonferroni correction.

- [x] A benchmark study s multi-omics data fusion methods for cancer: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02739-2
    - Много разных моделей, пока ничего не поняла. Но есть обзоры и много ссылок, может пригодится текст
    - Кода нет

- [x] MUON: multimodal omics analysis framework: https://link.springer.com/article/10.1186/s13059-021-02577-8
    - Библиотека python для визуализации, анализа
    - Пример анализа на факторов: https://muon-tutorials.readthedocs.io/en/latest/CLL.html?highlight=factor, https://muon-tutorials.readthedocs.io/en/latest/mefisto/2-MEFISTO-microbiome.html?highlight=factor

- [x] Capsule Network Based Modeling of Multi-omics Data for Discovery of Breast Cancer-Related Genes: https://ieeexplore.ieee.org/document/8684326
    - Задача классификации, использовали ML-метрики, сравнивали с классическими моделями
    - Данные и веса не доступны - ссылки не рабочие

- [x] Learning Cell-Type-Specific Gene Regulation Mechanisms by Multi-Attention Based Deep Learning With Regulatory Latent Space: https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2020.00869/full
    - To begin, separate models embed histone marks, DNA methylation, and transcription factors into a regulatory latent space. First, histone marks are embedded into the latent space by a Convolutional Neural Network (CNN) followed by a Bi-directional Long Short-Term Memory (LSTM) network with attention. Second, DNA methylation is vectorized by a Dynamic Bi-directional LSTM with attention. Lastly, a Self-Attention Network (SAN) embeds the transcription factors. After embedding features in three vectors, a Multi-Attention network combines these vectors to predict whether a gene would be highly expressed or lowly expressed.
    - Данные: Roadmap Epigenomics Projects
    - Код: https://github.com/pptnz/deeply-learning-regulatory-latent-space

- [x] Survival Analysis concepts: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394262/#:~:text=Survival%20analysis%20is%20a%20collection,interest%20will%20often%20be%20unknown
    - Survival analysis is a collection of statistical procedures for data analysis where the outcome variable of interest is time until an event occurs. Because of censoring–the nonobservation of the event of interest after a period of follow-up–a proportion of the survival times of interest will often be unknown. It is assumed that those patients who are censored have the same survival prospects as those who continue to be followed, that is, the censoring is uninformative. Survival data are generally described and modelled in terms of two related functions, the survivor function and the hazard function. The survivor function represents the probability that an individual survives from the time of origin to some time beyond time t. It directly describes the survival experience of a study cohort, and is usually estimated by the KM method. The logrank test may be used to test for differences between survival curves for groups, such as treatment arms. The hazard function gives the instantaneous potential of having an event at a time, given survival up to that time. It is used primarily as a diagnostic tool or for specifying a mathematical model for survival analysis. In comparing treatments or prognostic groups in terms of survival, it is often necessary to adjust for patient-related factors that could potentially affect the survival time of a patient. Failure to adjust for confounders may result in spurious effects. Multivariate survival analysis, a form of multiple regression, provides a way of doing this adjustment, and is the subject the next paper in this series.

- [x] Deep Learning–Based Multi-Omics Integration Robustly Predicts Survival in Liver Cancer: https://aacrjournals.org/clincancerres/article/24/6/1248/475/Deep-Learning-Based-Multi-Omics-Integration
    - This DL-based model provides two optimal subgroups of patients with significant survival differences (P = 7.13e−6) and good model fitness [concordance index (C-index) = 0.68].
    - 5 других датасетов: We validated this multi-omics model on five external datasets of various omics types: LIRI-JP cohort (n = 230, C-index = 0.75), NCI cohort (n = 221, C-index = 0.67), Chinese cohort (n = 166, C-index = 0.69), E-TABM-36 cohort (n = 40, C-index = 0.77), and Hawaiian cohort (n = 27, C-index = 0.82).
    - Their benchmarking method: 
    In step 1, mRNA, DNA methylation, and miRNA features from the TCGA HCC cohort are stacked up /_note: concatenated_/ as input features for the autoencoder, a DL method  /_note: with 1 hidden layer, 50% dropout_/; then each of the new, transformed features in the bottleneck layer of the autoencoder is then subjected to single variate Cox-PH models to select the features associated with survival; then K-mean clustering is applied to samples represented by these features to identify survival-risk groups. In step 2, mRNA, methylation, and miRNA input features are ranked by ANOVA test F values, those features that are in common with the predicting dataset are selected, then the top features are used to build an SVM model(s) to predict the survival-risk labels of new datasets.

- [x] Stacked Autoencoder Based Multi-Omics Data Integration for Cancer Survival Prediction: https://arxiv.org/abs/2207.04878
    - 3 типа рака из TCGA, лучшие С-index в районе 0.6-0.7
    - Модель отдельно для каждого рака

- [x] Dealing with dimensionality: the application of machine learning to multi-omics data: https://academic.oup.com/bioinformatics/article/39/2/btad021/6986971
    - Ссылки на статьи новые
    - Attention mechanisms показали лучшие результаты (??)

 - [x] Multi-level attention graph neural network based on co-expression gene modules for disease diagnosis and prognosis: https://academic.oup.com/bioinformatics/article/38/8/2178/6528315
    - https://github.com/TencentAILabHealthcare/MLA-GNN
    - Графовая НС
    
- [x] An in-depth comparison of linear and non-linear joint embedding methods for bulk and single-cell multi-omics:
 https://academic.oup.com/bib/article/25/1/bbad416/7450271 
    - Оч свежая, Jan 2024, описания автоэнкодеров

### RGCCA
https://github.com/rgcca-factory/RGCCA - R

From paper Multiblock data analysis with the RGCCA package (https://cran.r-project.org/web/packages/RGCCA/vignettes/RGCCA.pdf):

For Python users, mvlearn (Perry, Mischler, Guo, Lee, Chang, Koul, Franz, Richard, Carmichael, Ablin et al. 2021) seems to be the most advanced Python module for multiview data. 
mvlearn offers a suite of algorithms for learning latent space embeddings and joint representations of views, 
including a limited version of the RGCCA framework, a kernel version of GCCA (Hardoon, Szed- mak, and Shawe-Taylor 2004; Bach and Jordan 2002), 
and deep CCA (Andrew, Arora, Bilmes, and Livescu 2013). Several other methods for dimensionality reduction and joint subspace learning are also included, 
such as multiview multidimensional scaling (Trendafilov 2010).


## Текущий статус / вопросы

1. Сформулировать цель исследования
Сравнить классические unsupervised методы / DL-методы jDR по предсказанию выживаемости (оставшееся время жизни)

2. Дедлайн для исследования для диплома
Конец марта

3. Литература на которой основываться
Методы DL для cancer survival prediction

#### ToDo:
    - [ ] добавить mean С-index
    - [ ] добавить UMAP
    - [x] найти понятную имплементацию автоэнкодера в статье, их скор, и либо изменить свой, либо сравнить или показать чем лучше - Liver TCGA
    - [ ] НС с self-attention
    - [ ] сделать предсказания без 2х клин фичей про метастазы
