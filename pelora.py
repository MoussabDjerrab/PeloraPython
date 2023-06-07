import numpy as np
#step 1

p = 100  
n = 50  
X = np.random.normal(0, 1, size=(p, n))

X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True) 

#step 2
def compute_score(x, y):
    indicies_of_zero = np.where(y == 0)[0]
    indicies_of_one = np.where(y == 1)[0]
    
    score = np.zeros(x.shape[0])
     # gene 1: X = [x11 x12 X13 ... X1n] Y = [0 1 0 ... 1]

    for zero_index in indicies_of_zero:
        for one_index in indicies_of_one:
            score += np.where(x[:, zero_index] >= x[:, one_index], 1, 0)
    
    return score

def compute_margin(x, y):
    rows, cols = x.shape
    
    indicies_of_zero = np.where(y == 0)[0]
    indicies_of_one = np.where(y == 1)[0]
    
    margin = np.zeros(rows)

      # gene 1: X = [x11 x12 X13 ... X1n] Y = [0 1 0 ... 1]
    for i in range(rows):
        min_value = np.min(x[i, indicies_of_one])
        max_value = np.max(x[i, indicies_of_zero])
        margin[i] = min_value - max_value
    
    return margin
