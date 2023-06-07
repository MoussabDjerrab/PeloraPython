import numpy as np
#STEP 1
# Declare the dimensions of the expression matrix
p = 46  # Number of genes
n = 2   #(tissue types)

X = np.random.rand(p, n)

# Calculate the mean and standard deviation of each column (observation) in X
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

# Standardize the columns of X to have zero mean and unit variance
X_standardized = (X - mean) / std

# Verify the shape and statistics of the standardized matrix
print("Standardized X shape:", X_standardized.shape)
print("Standardized X mean:", np.mean(X_standardized, axis=0))
print("Standardized X standard deviation:", np.std(X_standardized, axis=0))

#------------------------------------------------------------------#
#STEP 2
def Calculate_score(x, y):
    rows, cols = x.shape
    y_rows = y.shape[0]
    
    indicies_of_zero = [k for k in range(y_rows) if y[k] == 0]
    indicies_of_one = [k for k in range(y_rows) if y[k] == 1]
    
    score = [0 for _ in range(rows)]
    
     #gene 1: X = [x11 x12 X13 ... Xpn] Y = [0 1 0 ... 1]
    # p:variables in colomn , n cases in rows 
    for i in range(rows):
        for zero_index in indicies_of_zero:
            for one_index in indicies_of_one:
                score[i] += 1 if x[i, zero_index] >= x[i, one_index] else 0
    
    return np.array(score)

def Calculate_score(x, y):
    rows, cols = x.shape
    y_rows = y.shape[0]
    
    indicies_of_zero = [k for k in range(y_rows) if y[k] == 0]
    indicies_of_one = [k for k in range(y_rows) if y[k] == 1]
    
    score = [0 for _ in range(rows)]
    
    for i in range(rows):  # Iterate over genes (rows)
        for zero_index in indicies_of_zero:
            for one_index in indicies_of_one:
                score[i] += 1 if x[i, zero_index] >= x[i, one_index] else 0
    
    # Calculate smax
    smax = max(score)
    
    flipped_x = np.copy(x)  # to store the flipped values we need to create a copy of x

    #flipped function 
    for i in range(rows): 
        if score[i] > smax / 2:
            flipped_x[i] = -flipped_x[i]  # Flip the sign of the expression vector
    
    return flipped_x
#----------------------------------------------------------------------#
# STEP THREE
def Calculate_score(x, y):
    rows, cols = x.shape
    y_rows = y.shape[0]
    
    indicies_of_zero = [k for k in range(y_rows) if y[k] == 0]
    indicies_of_one = [k for k in range(y_rows) if y[k] == 1]
    
    margin = [0 for _ in range(rows)]
    
    # gene 1: X = [x11 x12 X13 ... X1n]-->"our input" /Y = [0 1 0 ... 1]-->"our label"
    for i in range(rows):
        margin[i] = np.min(x[i, indicies_of_one]) - np.max(x[i, indicies_of_zero])
    
    return np.array(margin)

#------------------------------------------------------------------------#
 #Step Four
 
    #we need to find the gene with the lowest score and largest margin
def find_initial_gene(data, scores, margins):
    #we need to find the gene which has the lowest score
    min_score_idx = np.argmin(scores)
    
    # in case of the multiple genes have the lowest score,
    # we need find the gene with the largest margin
    genes_with_min_score = np.where(scores == scores[min_score_idx])[0]
    if len(genes_with_min_score) > 1:
        max_margin_idx = np.argmax(margins[genes_with_min_score])
        initial_gene_idx = genes_with_min_score[max_margin_idx]
    else:
        initial_gene_idx = genes_with_min_score[0]
    
    return data[initial_gene_idx]


# we  Calculate the average expression of genes in the initial cluster
def calculate_initial_cluster_mean(data):
    return np.mean(data, axis=0)


data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  
scores = np.array([0.5, 0.2, 0.3])  
margins = np.array([0.1, 0.4, 0.2]) 

# Step (a): Find the gene with the lowest score and largest margin
initial_gene = find_initial_gene(data, scores, margins)

# Step (b): Calculate the initial cluster mean
initial_cluster_mean = calculate_initial_cluster_mean(data)



