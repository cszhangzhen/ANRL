# Useful functions

import networkx as nx
import numpy as np
import collections
import random
import tensorflow as tf


def read_graph(FLAGS, edgeFile):
    print "loading graph..."

    if FLAGS.weighted:
        G = nx.read_edgelist(edgeFile, nodetype=int, data=(
            ('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(edgeFile, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not FLAGS.directed:
        G = G.to_undirected()

    return G


def read_edgelist(inputFileName):
    f = open(inputFileName, "r")
    lines = f.readlines()
    f.close()

    edgelist = []
    for line in lines:
        l = line.strip("\n\r").split(" ")
        edge = (int(l[0]), int(l[1]))
        edgelist.append(edge)
    return edgelist


def read_feature(inputFileName):
    f = open(inputFileName, "r")
    lines = f.readlines()
    f.close()

    features = []
    for line in lines[1:]:
        l = line.strip("\n\r").split(" ")
        features.append(l)
    features = np.array(features, dtype=np.float32)
    features[features > 0] = 1.0  # feature binarization

    return features


def write_embedding(embedding_result, outputFileName):
    f = open(outputFileName, "w")
    N, dims = embedding_result.shape

    for i in range(N):
        s = ""
        for j in range(dims):
            if j == 0:
                s = str(i) + "," + str(embedding_result[i, j])
            else:
                s = s + "," + str(embedding_result[i, j])
        f.writelines(s + "\n")
    f.close()
