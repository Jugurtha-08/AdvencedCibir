import numpy as np
from scipy.spatial import distance

 
def manhattan(v1, v2):
    """ cette fen calcul la distance de Manhattan/cityblock
    distance v1  of v2 same size
     
    v1 (list) : array of first object
    v2 (list) : array of second object
    
    """
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sum(np.abs(v1-v2))
    return dist

def euclidean(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sqrt(np.sum(v1-v2)**2)
    return dist

def chebyshev(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.max(np.abs(v1-v2))
    return dist

def canberra(v1, v2):
    return distance.canberra(v1, v2)

def retrieve_similar_image(features_db,query_features,distance,num_result):
    distances=[]
    for instance in features_db:

        features, label, img_path= instance[:-2],instance[-2],instance[-1]
        if distance=='manhattan':
            dist=manhattan(query_features,features)
        elif distance=='euclidean':
            dist=euclidean(query_features,features)
        elif distance=='chebyshev':
            dist=chebyshev(query_features,features)
        elif distance=='canberra':
            dist=canberra(query_features,features)
        distances.append((img_path, dist, label))
    distances.sort(key=lambda x: x[1])
    return distances[:num_result]