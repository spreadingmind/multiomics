## Project info
1. `tcga_preprocess.ipynb`

Preprocessing of 2 TCGA datasets: breast and kidney. In the result we get data for downstream analysis for dimemtionality reduction algorithms.

2. `tcga_breast.ipynb`

Deriving factors from statistical models and building cancer survival regressor.



## Dimentionality Reduction Algorithms info

## Relevant papers and tools

- [ ] Benchmarking joint multi-omics dimensionality reduction approaches for the study of cancer: https://www.nature.com/articles/s41467-020-20430-7

- [ ] BENCHMARKING OF JOINT DIMENSIONALITY REDUCTION METHODS FOR THE ANALYSIS OF MULTI-OMICS DATA IN CANCER: https://libstore.ugent.be/fulltxt/RUG01/003/008/173/RUG01-003008173_2021_0001_AC.pdf

- [ ] A benchmark study s multi-omics data fusion methods for cancer: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02739-2

- [ ] MUON: multimodal omics analysis framework: https://link.springer.com/article/10.1186/s13059-021-02577-8

- [ ] Deep Learning–Based Multi-Omics Integration Robustly Predicts Survival in Liver Cancer: https://aacrjournals.org/clincancerres/article/24/6/1248/475/Deep-Learning-Based-Multi-Omics-Integration

- [ ] Multi-omics integration method based on attention deep learning network for biomedical data classification: https://www.sciencedirect.com/science/article/abs/pii/S0169260723000445

- [ ] Dealing with dimensionality: the application of machine learning to multi-omics data: https://academic.oup.com/bioinformatics/article/39/2/btad021/6986971

- [ ] An in-depth comparison of linear and non-linear joint embedding methods for bulk and single-cell multi-omics:
 https://academic.oup.com/bib/article/25/1/bbad416/7450271 


### RGCCA
https://github.com/rgcca-factory/RGCCA - R

From paper Multiblock data analysis with the RGCCA package (https://cran.r-project.org/web/packages/RGCCA/vignettes/RGCCA.pdf):

For Python users, mvlearn (Perry, Mischler, Guo, Lee, Chang, Koul, Franz, Richard, Carmichael, Ablin et al. 2021) seems to be the most advanced Python module for multiview data. 
mvlearn offers a suite of algorithms for learning latent space embeddings and joint representations of views, 
including a limited version of the RGCCA framework, a kernel version of GCCA (Hardoon, Szed- mak, and Shawe-Taylor 2004; Bach and Jordan 2002), 
and deep CCA (Andrew, Arora, Bilmes, and Livescu 2013). Several other methods for dimensionality reduction and joint subspace learning are also included, 
such as multiview multidimensional scaling (Trendafilov 2010).


## Текущий статус / вопросы

1. Есть ли точно уверенность что эта проблема актуальная?
WIP: в процессе обзор статей

2. Сформулировать цель исследования
Сравнить классические unsupervised методы / DL-методы jDR по предсказанию выживаемости (оставшееся время жизни)

3. Фиксируем ли текущий пайплайн анализа?
Поменять на регрессию

4. Дедлайн для исследования для диплома
Конец марта

5. Литература на которой основываться
Методы DL для cancer survival prediction
