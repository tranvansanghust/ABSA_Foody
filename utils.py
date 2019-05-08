import numpy as np

def shuffle(X, y):
    '''Shuffling coressponding 2 matrixes X and y
    Args:
        X: numpy array M*n
        y: numpy array M*k
    
    Return:

    '''
    l = len(y)
    idx = np.arange(l)
    np.random.shuffle(idx)

    return np.squeeze(X[idx]), np.squeeze(y[idx])