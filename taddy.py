import numpy as np

def variancedist(W, S):
    try:
        R = np.linalg.inv(S.T @ S)
    except:
        R = np.linalg.inv(S.T @ S + np.eye(S.shape[1]) * 0.001)
    D = np.sum( (W @ R) * W, axis=1)
    return D

def dopt(topicmat, k):
    index = [np.random.choice(topicmat.shape[0])]
    rows = np.array(range(topicmat.shape[0]))
    rows = np.delete(rows, index)
    S = topicmat[index,:]
    W = topicmat[~np.isin(range(topicmat.shape[0]), index)]
    # print(S.shape, W.shape)

    while len(index) < k:
        i = np.argmax(variancedist(W, S))
        S = np.vstack((S, W[i,:]))
        W = np.delete(W, i, axis=0)
        index.append(rows[i])
        rows = np.delete(rows, i)

    return index
