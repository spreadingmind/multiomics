## Project info
1. `tcga_preprocess.ipynb`

Preprocessing of 2 TCGA datasets: breast and kidney. In the result we get common data for downstream analysis with dimemtionality reduction algorithms.

2. `tcga_breast_kidney_mofa.ipynb`

Deriving factors from MOFA model and building cancer type binary classifier.

3. `tcga_breast_kidney_rgcca.ipynb`

Deriving factors from RGCCA model and building cancer type binary classifier.


## Dimentionality Reduction Algorithms info

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