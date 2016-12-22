# Final project for E6893 Big Data Analytics

## Group Index:  201605-96

### Group member: 
* **Ruixiong Shi  rs3569**
* **Yuchen   Shi  ys2901**

### Training Network:

* model.py: model strucutre written and auxiliary function
* train.ipynb: Notebook for training the network
* model.pickle.gz: Pickled file network parameters

### Visualization

* pyspark_visual.ipynb: Visualize dataset by spark
* read.py: auxiliary function for reading hdf5 file into spark rdd file
* TSNE2D.ipynb: Visualize embedding space in 2D with T-SNE
* tf-embedding-visualization: Visualize embedding space in 3D with T-SNE

### Generating new fonts

* embedding_perturb_multinormal.ipynb: Creating new fonts by random perturbation and sampling from multivariate normal distribution of original embedding space
* random_perturb: output character created by random perturbation
* multivariate_norm: output character created by sampling from multivariate normal distribution

* interpolation.ipynb: Creating new fonts by interpolating embedding space

### Download Dataset

```
$ wget https://s3.amazonaws.com/erikbern/fonts.hdf5  
```

### Youtube Video

[Please click here](https://www.youtube.com/watch?v=nPQobbPFBgQ)

