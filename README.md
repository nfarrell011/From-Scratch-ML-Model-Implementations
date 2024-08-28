# Machine Learning Model Implementations  
___

### Overview
This folder contains modules of various machine learning models. Within each module is a class, i.e., ```model.py``` which contains a model class, i.e., the model implementation.  

Each model is implemented using only ```pandas```, ```matplotlib```, and basic ```numpy``` operations, i.e., no signnificant use of ```numpy``` functions.

Each model contain methods that create various plots and compute various metrics that are associated with the given model. For example, ```KMeans``` contains an implementation of ```silhouette_score``` and a plot that demonstrates the behavior of ```inertia``` w.r.t ```k```.

### Models  
* [KMeans](KMeans/README.md)
* [DBSCAN](DBSCAN/README.md)

### Metrics
* [Distance Metrics](distance_metrics/README.md)