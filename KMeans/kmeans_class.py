"""  
Nelson Farrell
08-22-2024
Northeastern University

This file contains an implementation of KMeans
"""
###############################################################################################################################################
#                                                               Packages                                                                      #
###############################################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import distance metrics class
from distance_metrics.distance_metrics_class import DistanceMetrics
###############################################################################################################################################
#                                                                   Class                                                                     #
###############################################################################################################################################
class KMeans:
    """  
    KMeans unsupervised clustering

        Attributes:
            * data: (np.array) - Data to perform clustering on.
            * k: (int) - The number of cluster centers.
            * random_seed: (int) - Set for reproducibility. Used in generating initial random cluster centers.
            * max_iters: (int) - The max number of interations.
            * clusters: (dict) - Container for current cluster assignments.
            * silhouette_scores: (dict) - Container for the silhouette scores of each point.
            * distance_metric: (callable) - Distance metric function
            * simplified_silhouette_scores - (dict) - Container for the simplified silhouette scores.
            * predictions: (dict) - Container for predictions on new data from fit model.
            * intertia: (float) - The intertia, sum of the distance from each point the its cluster center.

        Methods:
            * fit
            * predict
            * euclidean -- NO LONGER USED
            * find_best_k
            * get_interia
            * get_silhouette_score
            * get_simplified_silhouette_score
            * plot_silhouette_score
    """
    def __init__(self, data:np.array, max_iters:int = 15, random_seed:int = None, distance_function:callable = None) -> None:
        """  
        Model initializer

        Args:
            * data: (np.array) - Data to perform clustering on.
            * random_seed: (int) - (Optional) - Set for reproducibility. Used in generating initial random cluster centers. Default = None.
            * max_iters: (int) - (Optional) - The max the number of interations. Default = None.
            * distance_metric: (callable) - (Optional) - Distance metric function. Default = Euclidean.
        """
        self.data = data
        self.k = None
        self.random_seed = random_seed
        self.max_iters = max_iters
        self.clusters = None
        self.silhouette_scores = {
                                    "point": [],
                                    "cluster": [],
                                    "silhouette_score": []
                                 }
        self.simplified_silhouette_scores = {
                                                "point": [],
                                                "cluster": [],
                                                "silhouette_score": []
                                            }
        self.silhouette_coefficient = None
        self.simplified_silhouette_coefficient = None
        self.predictions =  {
                              "point": [],
                              "cluster": []
                            }
        self.inertia = None
        if random_seed is not None:
            np.random.seed(random_seed) # set numpy random seed

        # Set the distance function, if one is provided
        self.distance_metric = distance_function if distance_function else DistanceMetrics().euclidean

###############################################################################################################################################
#                                                                  KMeans                                                                     #
###############################################################################################################################################
    def set_cluster_centers(self) -> None:
        """ 
        Initializes cluster centers
        """
        # Container for clusters
        clusters = {}

        # Used to ensure initial cluster centers are within range of the data
        data_min = np.min(self.data, axis=0)
        data_max = np.max(self.data, axis=0)

        # Initialize "k" cluster centers
        for idx in range(self.k):

            # Generate randomized cluster centers within the data range
            center = np.random.uniform(data_min, data_max, size=self.data.shape[1])

            # Container for points within a cluster, currently empty
            points = []

            # Cluster container and tracker
            cluster = {
                'center': center,
                'points': points
            }

            # Update cluster container
            clusters[idx] = cluster

        # Set to clusters attribute
        self.clusters = clusters

    def assign_cluster(self) -> None:
        """  
        Assigns points to a cluster by calculating the distance from each point to each cluster center and 
        assigning the point to the cluster with the closest cluster center.
        """
        # Clear previous points
        for cluster in self.clusters.values():
            cluster['points'] = []
        
        # Interate over each point in the data
        for idx in range(self.data.shape[0]):

            # Extract individual point
            point_i = self.data[idx]

            # Calculate the distance to each cluster center
            dist = [self.distance_metric(point_i, self.clusters[i]["center"]) for i in range(self.k)]

            # Extract the index of the clostest cluster center
            curr_cluster = np.argmin(dist)

            # Add to point the cluster
            self.clusters[curr_cluster]["points"].append(point_i)

    def update_cluster_centers(self) -> None:
        """ 
        Update the cluster centers by calculating the column mean of all the points in the cluster.
        The mean of each column will become the elements of the new cluster center.
        """
        # Iterate over each of the "k" clusters
        for i in range(self.k):

            # Extract the points in a cluster
            points = np.array(self.clusters[i]["points"])

            # Check if there are points in the cluster
            if points.shape[0] > 0:

                # Calcualte the column means; the new cluster center
                new_center = points.mean(axis = 0)

                # Update cluster center
                self.clusters[i]["center"] = new_center

    def euclidean(self, p_1:np.array, p_2:np.array) -> float:
        """  
        Euclidean distance metric

        Args:
            * p_1: (np.array) - A vector of coordinates.
            * p_2: (np.array) - A vector of coordinates.

        Returns:
            * (float) - The Euclidean distance between the two points.
        """
        return np.sqrt(np.sum((p_1 - p_2) ** 2))
    
    def fit(self, k:int, show_clusters:bool = False) -> None:
        """  
        Fits Kmeans to the data using the above functions.
        """
        # Set k
        self.k = k

        # Initialize the cluster centers
        self.set_cluster_centers()

        # Perform iterative updates; "max_iters" iterations
        for i in range(self.max_iters):

            # Assign points to a cluster
            self.assign_cluster()

            if show_clusters:
                
                # Show the clusters
                self.show_clusters()

            # Update cluser centers.
            self.update_cluster_centers()

    def predict(self, points:list) -> None:
        """ 
        Predict cluster of new points using fit model.
        """
        # Iterate over the points
        for i in range(points.shape[0]):

            # Extract individual point
            point = points[i]

            # Container for distances to each cluster center.
            dist_list = []

            # Iterate over the cluster center.
            for j in range(self.k):

                # Calculate the distance to a cluster center.
                dist = self.distance_metric(point, self.clusters[j]["center"])

                # Update distance to cluster center list.
                dist_list.append(dist)
            
            # Get the index of the minimum distance, the cluster label.
            pred = np.argmin(dist_list)

            # Update predictions dict.
            self.predictions["point"].append(point)
            self.predictions["cluster"].append(pred)

###############################################################################################################################################
#                                                             Indices and Plots                                                               #
###############################################################################################################################################
    def find_best_k(self, k_list:list = [2, 3, 4, 5, 6, 7, 8, 9, 10]) -> None:
        """  
        Instantiates KMEANs for each value of k, tracking interia and plotting results. The elbow in the resultant
        plot is the best value to use as k when instaniating final model.

        Args:
            * k_list: (list) - (Optional) List of k values to check. Default =  [2, 3, 4, 5, 6, 7, 8, 9, 10]
        """
        # Container for results; k values, and the resultant intertia score
        results_dict = {
                        "k": [],
                        "inertia": []
                       }
        # Iterate over the k values
        for k in k_list:

            # Fit the model with k value
            self.fit(k)
            self.get_intertia()
            
            # Update results dict
            results_dict["k"].append(k)
            results_dict["inertia"].append(self.inertia)

        # Create results df
        df = pd.DataFrame(results_dict)

        # Generate Figure
        plt.figure(figsize = (10, 10))
        plt.plot(df["k"], df["inertia"], marker = "o", linestyle = "-")
        plt.title("Elbow Method to Find Optimal k", weight = "bold")
        plt.xlabel("Number of Clusters: (k)", weight = "bold")
        plt.ylabel("Inertia", weight = "bold")


    def show_clusters(self) -> None:
        """  
        Displays the current cluster centeres and the cluster assignments if data is 2-dimensional.
        """
        # Check the dims of the data
        if self.data.shape[1] == 2:

            # List of colors for each cluster
            colors = ["deepskyblue", "yellow", "r", "g", "b"]

            # Iterate over the current cluster assignments
            for i in range(len(self.clusters)):

                # Extract the points in a cluster
                points = np.array(self.clusters[i]["points"])

                # Extract cluster center
                center = self.clusters[i]["center"]

                # Check if there points to plot in the cluster
                if points.size > 0:

                    # Plot the points
                    plt.scatter(points[:, 0], points[:, 1], c=colors[i], label=f'Cluster {i}')
                
                # Plot the center
                plt.scatter(center[0], center[1], c="black", marker="x", s=100)
            plt.legend()
            plt.show()
        else:
            return

    def get_intertia(self) -> float:
        """
        Computes the interia metric of a clustering outcome.
        """
        inertia = 0
        for i in range(self.k):
            center = self.clusters[i]["center"]
            points = np.array(self.clusters[i]["points"])
            for j in range(points.shape[0]):
                point = points[j]
                dist = self.distance_metric(center, point)
                inertia += dist
        self.inertia = inertia
        return inertia
    
    def get_silhouette_score(self) -> float:
        """  
        Computes the silhouette score of a clustering outcome.
        This in an O(n^2) operation! More costly than KMeans itself.
        Simplified Silhouette is also implemented to avoid this operation when needed. 
        """
        # Container for all the silhouette scores for each point
        all_silhouette_scores = []
        
        # Iterate over each cluster
        for i in range(self.k):

            # Extract all the points a the cluster
            in_points = np.array(self.clusters[i]["points"])
            
            # Skip if there are no points or only one point in the cluster
            if len(in_points) <= 1:
                continue
            
            # Iterate over the points in the current cluster
            for j in range(len(in_points)):

                # Point of interest
                point_j = in_points[j]
                
                # Compute the average intra-cluster distance (a)
                a = np.mean([self.distance_metric(point_j, in_points[p]) for p in range(len(in_points)) if p != j])

                # Compute the average nearest-cluster distance (b)
                b = np.inf  # Initialize b as a large value
                
                # Iterate over the other clusters
                for k in range(self.k):
                    if i == k:
                        continue  # Skip the same cluster
                    
                    # Extract the points from the out cluster
                    out_points = np.array(self.clusters[k]["points"])
                    
                    if len(out_points) > 0:
                        dist_to_other_cluster = np.mean([self.distance_metric(point_j, out_points[q]) for q in range(len(out_points))])
                        b = min(b, dist_to_other_cluster)
                
                # Compute the silhouette score for point_j
                sil_score_j = (b - a) / max(a, b)

                # Save the sil score for each point
                self.silhouette_scores["point"].append(point_j)
                self.silhouette_scores["cluster"].append(i)
                self.silhouette_scores["silhouette_score"].append(sil_score_j)

                # Add score to silhouette scores list
                all_silhouette_scores.append(sil_score_j)
        
        # Compute the average silhouette score over all points; i.e. silhouette score
        silhouette_coefficient = np.mean(all_silhouette_scores) if all_silhouette_scores else 0

        # Save and return silhouette coefficient
        self.silhouette_coefficient = silhouette_coefficient
        return silhouette_coefficient
    
    def get_simplified_silhouette_score(self) -> float:
        """  
        Computes simplified silhouette score using centriods rather element wise distances.
        """
        # Containter for cluser silhouette scores
        cluster_silhouette_scores = []

        # Iterate over each cluster
        for i in range(self.k):

            # Container for all the silhouette scores for each point; in a cluster
            all_silhouette_scores = []

            # Extract the center
            in_cluster_center_point = self.clusters[i]["center"]

            # Extract the points in the cluster
            points = np.array(self.clusters[i]["points"])

            # Skip if there are no points or only one point
            if len(points) <= 1:
                continue

            # Iterate over the points in the current cluster
            for j in range(len(points)):

                # Point of interest
                point_j = points[j]

                # Compute the distance to the center
                a = self.distance_metric(in_cluster_center_point, point_j)

                # Set be to infinity
                b = np.inf

                # Iterate over the other clusters
                for k in range(self.k):
                    if i == k:
                        continue # Skip the same cluster
                    
                    # Extract the center of the out cluster
                    out_cluster_center_point = self.clusters[k]["center"]

                    # Extract the points from the out cluster
                    out_points = self.clusters[k]["points"]

                    # If there are no points, skip cluster
                    if len(out_points) == 0:
                        continue
                    
                    # Calculate distance to out cluster center
                    dist_to_other_cluster = self.distance_metric(out_cluster_center_point, point_j)

                    # Update b, the minimum distance seen
                    b = min(b, dist_to_other_cluster)
                
                # Compute silhouette score for point_j
                sil_score_j = (b - a) / max(a, b)

                # Add point_j silhouette score to the list of in cluser silhouette scores
                all_silhouette_scores.append(sil_score_j)

                # Update results container
                self.simplified_silhouette_scores["point"].append(point_j)
                self.simplified_silhouette_scores["cluster"].append(i)
                self.simplified_silhouette_scores["silhouette_score"].append(sil_score_j)

            # Compute the mean of silhouette scores for a given cluster
            cluster_sil_score = np.mean(all_silhouette_scores)

            # Add cluster silhouette to the list of cluster sil scores
            cluster_silhouette_scores.append(cluster_sil_score)

        # Compute overall silhouette score
        silhouette_score = max(cluster_silhouette_scores)

        # Update container
        self.simplified_silhouette_coefficient = silhouette_score

        # Return the max of the cluser silhouette scores as the silhouette score
        return silhouette_score
        

    def plot_silhouette_scores(self, use_simplified = False) -> None:
        """  
        Plots the silhouette scores
        """
        # Use simplified silhouette scores if indicated
        if use_simplified:

            # Put results in a dataframe
            df = pd.DataFrame(self.simplified_silhouette_scores)

        # Put results in a dataframe
        df = pd.DataFrame(self.silhouette_scores)

        # If there are only 2 features the points will be plotted also
        if self.data.shape[1] == 2:

            # Parse the points into x and y coordinates
            df[["x", "y"]] = df["point"].apply(lambda p: pd.Series([p[0], p[1]]))

            # Set a y_lower; used for spacing
            y_lower = 10

            # Create the figure
            fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize = (20,10))
            ax_1.set_xlim([-0.1, 1])
            ax_1.set_ylim([0, len(df) + (3 + 1) * 10])

            # Iterate over the clusters
            for i in range(self.k):

                # Extract all the points in cluster i
                cluster_df = df[df["cluster"] == i].copy()

                # Sort the silhouette scores
                cluster_df.sort_values(by = "silhouette_score", inplace = True)

                # Get the number points in the cluster
                size_cluster = cluster_df.shape[0]

                # Set upper bound
                y_upper = y_lower + size_cluster

                # Set the color for teh cluster
                color = cm.nipy_spectral(float(i) / self.k)

                # Generate plot
                ax_1.fill_betweenx(np.arange(y_lower, y_upper),
                                0,
                                cluster_df["silhouette_score"],
                                facecolor = color,
                                edgecolor = color,
                                alpha = 0.7)
                
                # Add the cluster label text to the left of the plot
                ax_1.text(-0.05, y_lower + 0.5 * size_cluster, str(i))

                # Adjust the lower bound for next cluster
                y_lower = y_upper + 10

            # Add a vertical line for the overall silhouette score
            ax_1.axvline(x = self.silhouette_coefficient, color = "red", linestyle = "--")

            # Set the title and axis
            ax_1.set_title("Cluster Silhouette Scores", weight = "bold", fontsize = 18)
            ax_1.set_xlabel("Silhouette Coefficient", weight = "bold")
            ax_1.set_ylabel("Cluster Labels", weight = "bold")
            ax_1.set_yticks([])

            # Set the colors for each point
            colors = cm.nipy_spectral(df["cluster"].astype(float) / self.k)

            # Plot each point delineated by color
            ax_2.scatter(df["x"], df["y"], marker = ".", s = 30, c = colors)

            # Plot the cluter centers red
            for i in range(self.k):
                center = self.clusters[i]["center"]
                ax_2.scatter(center[0],
                            center[1],
                            marker = "o",
                            color = "r",
                            alpha = 1)

            # Set the title and axes
            ax_2.set_title("Clusters and Cluster Centers", weight = "bold", fontsize = 18)
            ax_2.set_xlabel("Feature 1", weight = "bold")
            ax_2.set_xlabel("Feature 2", weight = "bold")

        else:
            # Set a y_lower; used for spacing
            y_lower = 10

            # Create the figure
            plt.figure(figsize = (8,8))
            plt.xlim([-0.1, 1])
            plt.ylim([0, len(df) + (3 + 1) * 10])

            # Iterate over the clusters
            for i in range(self.k):

                # Extract all the points in cluster i
                cluster_df = df[df["cluster"] == i].copy()

                # Sort the silhouette scores
                cluster_df.sort_values(by = "silhouette_score", inplace = True)

                # Get the number points in the cluster
                size_cluster = cluster_df.shape[0]

                # Set upper bound
                y_upper = y_lower + size_cluster

                # Set the color for teh cluster
                color = cm.nipy_spectral(float(i) / self.k)

                # Generate plot
                plt.fill_betweenx(np.arange(y_lower, y_upper),
                                0,
                                cluster_df["silhouette_score"],
                                facecolor = color,
                                edgecolor = color,
                                alpha = 0.7)
                
                # Add the cluster label text to the left of the plot
                plt.text(-0.05, y_lower + 0.5 * size_cluster, str(i))

                # Adjust the lower bound for next cluster
                y_lower = y_upper + 10

            # Add a vertical line for the overall silhouette score
            plt.axvline(x = self.silhouette_coefficient, color = "red", linestyle = "--")

            # Set the title and axis
            plt.title("Cluster Silhouette Scores", weight = "bold", fontsize = 18)
            plt.xlabel("Silhouette Coefficient", weight = "bold")
            plt.ylabel("Cluster Labels", weight = "bold")
            plt.yticks([])
###############################################################################################################################################
#                                                                   End                                                                       #
###############################################################################################################################################
if __name__ == "__main__":
    pass