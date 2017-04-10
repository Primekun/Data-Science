#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:39:26 2017

@author: neelabhpant
"""

from __future__ import division # want 3 / 2 == 1.5
import re, math, random # regexes, math functions, random numbers
import matplotlib.pyplot as plt # pyplot
from collections import defaultdict, Counter
from functools import partial

def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors):
    return reduce(vector_add, vectors)

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

def vector_mean(vectors):
    length = len(vectors)
    sum_of_vectors = vector_sum(vectors)
    return scalar_multiply(1/length, sum_of_vectors)

def dot(v,w):
    prod = [v_i * w_i for v_i, w_i in zip(v, w)]
    return sum(prod)

def sum_of_squares(v):
    return dot(v,v)

def magnitude(v):
    return(math.sqrt(sum_of_squares(v)))

def squared_distance(v,w):
    return sum_of_squares(vector_subtract(v,w))

def distance(v,w):
    return math.sqrt(squared_distance(v,w))

def shape(A):
    nr = len(A)
    nc = len(A[0])
    return nr, nc

def get_row(A, i):
    return A[i]

def get_column(A, j):
    return [A_i[j] for A_i in A]

def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)]
    
def is_diagonal(i, j):
    return 1 if i==j else 0

def matrix_add(v, w):
    if shape(v) != shape(w):
        raise ArithemeticError("cannot add matrrices of different shapes")
    
    num_row, num_col = shape(v)
    def entry_func(i,j): return v[i][j] + w[i][j]
    
    return make_matrix(num_row, num_col, entry_func)

friendships = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # user 0
               [1, 0, 1, 1, 0, 0, 0, 0, 0, 0], # user 1
               [1, 1, 0, 1, 0, 0, 0, 0, 0, 0], # user 2
               [0, 1, 1, 0, 1, 0, 0, 0, 0, 0], # user 3
               [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # user 4
               [0, 0, 0, 0, 1, 0, 1, 1, 0, 0], # user 5
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 6
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 7
               [0, 0, 0, 0, 0, 0, 1, 1, 0, 1], # user 8
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] # user 9

# find who all are friends with user 5

friend_of_five = []
for i, j in enumerate(friendships[5]):
    if j:
        friend_of_five.append(i)


    
def mean(v):
    return (sum(v)/len(v))

def de_mean(v):
    v_bar = mean(v)
    return [v_i - v_bar for v_i in v]

def variance(v):
    n = len(v)
    dist_from_mean = de_mean(v)
    sum_sq_diff = sum_of_squares(dist_from_mean)
    return (sum_sq_diff / n - 1)

def standard_deviation(v):
    return math.sqrt(variance(v))

def covariance(v, w):
    n = len(v)
    return dot(de_mean(v), de_mean(w)) / (n-1)

def correlation(v, w):
    std_v = standard_deviation(v)
    std_w = standard_deviation(w)
    return covariance(v, w) / std_v / std_w

def covariance_matrix(X):
    mean_vec = vector_mean(X)
    (X - mean_vec)

def PCA(data):
    #Subtract the mean from each of the dimensions.
    #find Correlation matrix
    #find eigenvalue and eigenvector if Correlation matrix
    # sort the eigenvector according to the eigenvalues in descending order
    # Multiply the Eigenvectors chosen wiht the original data
    pass

    













