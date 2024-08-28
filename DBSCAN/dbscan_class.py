"""  
Nelson Farrell
08-22-2024
Northeastern University

This file contains an implementation of DBSCAN
"""
###############################################################################################################################################
#                                                               Packages                                                                      #
###############################################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import matplotlib.cm as cm

###############################################################################################################################################
#                                                                   Class                                                                     #
###############################################################################################################################################
class DBSCAN:
    """  
    DBSCAN unsupervised clustering

        Attributes:
            * data: (np.array) - Data to perform clustering on.
            * eps: (float) - The radius around the point used to define a core point; i.e., locate points that close to the point of interest.
            * min_points: (int) - The number of points that must be with eps of a point for the point in question to considered a core point.
            * results_dict: (dict) - Dictionary where the keys will be the points stored as tuples and the values will be the cluster labels.
            * core_indices: (list) - Index list of core points; points that have "min_points" within "eps" distance.

        Methods:
            * initialize_results_dict
            * euclidean
            * find_best_eps
            * fit
            * region_query
            * grow_cluster
    """
    def __init__(self, data:np.array) -> None:
        """  
        Model inititializer.

        Args:
            * data: (np.array) - Data to perform clustering on.
        """
        self.data = data
        self.eps = None
        self.min_points = None
        self.results_dict = {}
        self.core_indices = []

    def initialize_results_dict(self) -> None:
        """  
        Initializes results dict converting each point to a tuple and using the tuple as a
        dictionary key
        """
        # Iterate of all data points
        for i in range(self.data.shape[0]):

            # Convert to tuple; round is used to ensure the values will match
            point = tuple(np.round(self.data[i], decimals = 8))

            # Add the point to the dictionary
            self.results_dict[point] = 0 # 0 indicating unseen

    def euclidean(self, point_1, point_2) -> float:
        """
        Computes the euclidean distance between two points.

        Args:
            * p_1: (np.array) - A vector of coordinates.
            * p_2: (np.array) - A vector of coordinates.

        Returns:
            * (flaot) - The Euclidean distance between the two points.
        """
        return np.sqrt(np.sum((point_1 - point_2) ** 2))
    
    def find_best_eps(self, k_list:list = [3, 4, 5, 6, 7], plot:bool = True) -> None:
        """
        Finds the best eps (distance from a point used to identify core points) by using k-distance approach.
        Computes the distance from each to point to k-th point and plots the results in descending order; the elbow
        of the resultant graph indentfies the best eps for a given k.

        K will be used as the hyperparameter "min_points" when invoking fit.

        Line search around discovered eps for best results.

        Args:
            * k_list: (list) - (Optional) List of k-values to use in search. Default = [3,4,5,6,7]
            * plot: (bool) - (Optional) Indicate whether or not show the resultant plot. Default = True
        """
        # Container for the results
        results_dict = {}

        # Iterate over the elements of k_list; used as the k-th neighbor
        for k in k_list:

            # Stores distances to the k-th neighbor for all the points
            k_dist_list = []

            # Iterare over all the points
            for point in self.data:

                # Compute distance to from POI to all other points
                dist_list = [self.euclidean(point, x) for x in self.data]

                # Sort the distances so the k-th distance extracted
                dist_list.sort()

                # Extract k-th distance
                k_dist = dist_list[k + 1] # (k+1) because the point is included

                # Update the k-th distance list
                k_dist_list.append(k_dist)
            
            # Sort the k-th distnace list
            k_dist_list.sort()

            # Update dict; for each k we have a sorted k-th distance list
            results_dict[k] = k_dist_list

        # If plot is True; generate figure
        if plot:
            plt.figure(figsize=(10, 6))
            for i, k in enumerate(k_list):
                x_values = range(len(results_dict[k]))
                color = cm.nipy_spectral(i / len(k_list))  # Normalize i for colormap
                plt.plot(x_values, results_dict[k], color=color, label=f'k = {k}')
                plt.legend()
                plt.title(r"Sorted Distances to $K^{th}$ Point", weight = "bold")
                plt.xlabel("Dataset Points", weight = "bold")
                plt.ylabel("Euclidean Distance", weight = "bold")
            plt.show()


    def fit(self, eps:float, min_points:int) -> None:
        """
        Fit method to execute the DBSCAN algorithm.

        Args:
            * eps: (float) - The radius around the point used to define a core point; i.e., locate points that close to the point of interest.
            * min_points: (int) - The number of points that must be with eps of a point for the point in question to considered a core point.
        """
        # Set attributes
        self.eps = eps
        self.min_points = min_points

        # Initialize the results dict
        self.initialize_results_dict()

        # Initialize the cluster labels
        cluster_label = 0

        # Iterate over all the points in the data
        for i in range(self.data.shape[0]):

            # Extract the point of interest
            point = self.data[i]

            # Create tuple of POI for dict indexing
            point_tuple = tuple(np.round(self.data[i], decimals = 8))

            # Check if the point has already been seen; if it has continue
            if self.results_dict[point_tuple] != 0:
                continue

            # Find all the points within eps distance of the POI
            neighbor_points = self.region_query(point)

            # If the number of neighbor points is less than min points, label POI as noise (-1)
            if len(neighbor_points) < self.min_points:
                self.results_dict[point_tuple] = -1
            
            # If there are enough points near the POI, use the point as a cluster seed
            else:
                cluster_label += 1
                self.core_indices.append(i)
                self.grow_cluster(point, neighbor_points, cluster_label)


    def region_query(self, point:np.array) -> list:
        """
        Finds all points within a given distance to a point interest

        Args:
            * point: (np.array) - The current point of interest.

        Returns:
            * nieghbor_points: (list) - All the points within eps of the POI.
        """
        # Intialize container for indices of neighbor points
        neighbor_points = []

        # Iterate over all the points in the data
        for point_n_index in range(len(self.data)):

            # Compute distance from POI to point_n
            dist = self.euclidean(point, self.data[point_n_index])

            # If the distance is less than eps, add point_n_index to neighbor list
            if dist < self.eps:
                neighbor_points.append(point_n_index)

        # Return neighbor points
        return neighbor_points

    def grow_cluster(self, point:np.array, neighbor_points:list, cluster_label:int) -> None:
        """ 
        Expands the current cluster by calling grow_region and appending neighbor_points.

        Args:
            * point: (np.array) - The current point of interest.
            * nieghbor_points: (list) - List of points that are within the current cluster.
            * cluster_label: (int) - The label of the current cluster.

        Returns:
            * None
        """
        # Label the POI with the current cluster label
        self.results_dict[tuple(point)] = cluster_label

        # Intialize tracker for neighbor_points; use while because neighbor_points may grow in length
        i = 0

        # Iterate over all the points in the neighbor_points list
        while i < len(neighbor_points):
            
            # Extract the index of point from the neighbors list
            point_n_index = neighbor_points[i]

            # Extract the corrresponding point from the data
            point_n = self.data[point_n_index]

            # Generate point tuple for dict indexing
            point_n_tuple = tuple(np.round(point_n, decimals=8))

            # Check if point_n was previously labeled as noise; if it was it cannot grow the region, but can
            # be part of the cluster
            if self.results_dict[point_n_tuple] == -1:
                self.results_dict[point_n_tuple] = cluster_label # update label
            
            # Check if point_n has not been seen
            elif self.results_dict[point_n_tuple] == 0:
                self.results_dict[point_n_tuple] = cluster_label # update cluster label

                # Find all the neighbors of point_n
                point_n_neighbor_points = self.region_query(point_n)

                # If point_n has at least min_points it is a branch point
                if len(point_n_neighbor_points) > self.min_points:

                    # Add point_n neighors to the queue (FIFO) to be searched
                    neighbor_points += point_n_neighbor_points
            
            # Update tracker
            i += 1
###############################################################################################################################################
#                                                                   End                                                                       #
###############################################################################################################################################
if __name__ == "__main__":
    pass