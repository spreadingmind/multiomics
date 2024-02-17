## Project info
1. `tcga_preprocess.ipynb`

Preprocessing of 2 TCGA datasets: breast and kidney. In the result we get common data for downstream analysis with dimemtionality reduction algorithms.

2. `tcga_breast_kidney_mofa.ipynb`

Deriving factors from MOFA model and building cancer type binary classifier.

3. `tcga_breast_kidney_rgcca.ipynb`

Deriving factors from RGCCA model and building cancer type binary classifier.


## Dimentionality Reduction Algorithms info

## Relevant papers and tools

Benchmarking joint multi-omics dimensionality reduction approaches for the study of cancer: https://www.nature.com/articles/s41467-020-20430-7

BENCHMARKING OF JOINT DIMENSIONALITY REDUCTION METHODS FOR THE ANALYSIS OF MULTI-OMICS DATA IN CANCER: https://libstore.ugent.be/fulltxt/RUG01/003/008/173/RUG01-003008173_2021_0001_AC.pdf

A benchmark study of deep learning-based multi-omics data fusion methods for cancer: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02739-2

MUON: multimodal omics analysis framework: https://link.springer.com/article/10.1186/s13059-021-02577-8



### MSFA
https://github.com/rdevito/MSFA - R

### RGCCA
https://github.com/rgcca-factory/RGCCA - R

From paper Multiblock data analysis with the RGCCA package (https://cran.r-project.org/web/packages/RGCCA/vignettes/RGCCA.pdf):

For Python users, mvlearn (Perry, Mischler, Guo, Lee, Chang, Koul, Franz, Richard, Carmichael, Ablin et al. 2021) seems to be the most advanced Python module for multiview data. 
mvlearn offers a suite of algorithms for learning latent space embeddings and joint representations of views, 
including a limited version of the RGCCA framework, a kernel version of GCCA (Hardoon, Szed- mak, and Shawe-Taylor 2004; Bach and Jordan 2002), 
and deep CCA (Andrew, Arora, Bilmes, and Livescu 2013). Several other methods for dimensionality reduction and joint subspace learning are also included, 
such as multiview multidimensional scaling (Trendafilov 2010).


## Текущий статус / вопросы

1. Есть ли точно уверенность что эта проблема актуальная? Что нет сравнения этих алгоритмов? 
2. Сформулировать цель исследования
3. Фиксируем ли текущий пайплайн анализа? 
4. Что можно добавить для интерпретации результатов, биологических выводов
5. Составить четкий план исследования и как должен выглядеть результат
6. Дедлайн для исследования - конец марта
7. Литература на которой основываться
8. Как насчет взять ML методы
