"""  
Nelson Farrell
08-22-2024
Northeastern University

This file contains an implementation of a class of various distance metrics.
"""
###############################################################################################################################################
#                                                               Packages                                                                      #
###############################################################################################################################################
import numpy as np
###############################################################################################################################################
#                                                             Metrics Class                                                                   #
###############################################################################################################################################
class DistanceMetrics:
    """
    A class of various distance metric implementations.

    Attributes:
        * None
    
    Methods:
        * euclidean
        * manhattan
        * chebyshev
        * cosine_distance
        * SSD
    """
    def __init__(self) -> None:
        """
        Class initializer.
        """
        pass

    def euclidean(self, point_1:np.array, point_2:np.array) -> float:
        """ 
        Computes the Euclidean distance, L2 norm.

        Args:
            * point_1: (np.array) - Input vector
            * point_2: (np.array) - Input vector

        Returns:
            * dist: (float) - The distance between the two vectors.
        """
        dist = np.sqrt(np.sum((point_1 - point_2) ** 2))
        return dist

    def manhattan(self, point_1:np.array, point_2:np.array) -> float:
        """
        Computes the Manhattan distance, L1 norm.

        Args:
            * point_1: (np.array) - Input vector
            * point_2: (np.array) - Input vector

        Returns:
            * dist: (float) - The distance between the two vectors.
        """
        dist = np.sum(np.abs(point_1 - point_2))
        return dist

    def chebyshev(self, point_1:np.array, point_2:np.array) -> float:
        """
        Computes the Chebyshev distance. L-infinity norm.

        Args:
            * point_1: (np.array) - Input vector
            * point_2: (np.array) - Input vector

        Returns:
            * dist: (float) - The distance between the two vectors.
        """
        dist = np.max(np.abs(point_1 - point_2))
        return dist

    def cosine_distance(self, point_1:np.array, point_2:np.array, return_similarity:bool = False) -> float:
        """
        Computes the Cosine distance; the angle between two vectors.

        Args:
            * point_1: (np.array) - Input vector
            * point_2: (np.array) - Input vector

        Returns:
            * dist: (float) - The distance between the two vectors.
        """
        dot_product = np.dot(point_1, point_2)
        norm_point_1 = np.sqrt(np.dot(point_1, point_1))
        norm_point_2 = np.sqrt(np.dot(point_2, point_2))
        cosine_sim = dot_product / (norm_point_1 * norm_point_2)
        if return_similarity:
            return cosine_sim
        else:
            dist = 1 - cosine_sim
            return dist

    def SSD(self, point_1:np.array, point_2:np.array) -> float:
        """
        Computes the sum of squared distance.

        Args:
            * point_1: (np.array) - Input vector
            * point_2: (np.array) - Input vector

        Returns:
            * dist: (float) - The distance between the two vectors.
        """
        dist = np.sum((point_1 - point_2) ** 2)
        return dist
    
###############################################################################################################################################
#                                                                  End                                                                        #
###############################################################################################################################################
if __name__=="__main__":
    pass