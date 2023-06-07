import numpy as np
 # Step 1
 
def generate_expression_matrix(p, n):
# Generate random expression matrix with zero mean and unit variance
    x = np.random.randn(p, n)

# Calculate the mean and standard deviation of each column
    column_means = np.mean(x, axis=0)
    column_std = np.std(x, axis=0)

# Subtract the mean and divide by the standard deviation for each column
    X_normalized = (x - column_means) / column_std

    return X_normalized
p = 200  # Number of genes
n = 100  # Number of observations


X_normalized = generate_expression_matrix(p, n)
print(X_normalized)

 # Step 2
x= np.array()

# Calculate the score for each gene

def calculate_score(data_set1, data_set2):
    score = 0
    for value1 in data_set1:
        for value2 in data_set2:
            if value1 >= value2:
                score += 1 
    return score
data_set1 = X_normalized[:, :50]  # First 50 samples from X_normalized
data_set2 = X_normalized[:, 50:]   # Last 50 samples from X_normalized

# Calculate Smax
n0 = data_set1.shape[1]  # Number of samples in data_set1
n1 = data_set2.shape[1]  # Number of samples in data_set2

Smax = n0 * n1 * score  # Calculate Smax

print("Smax:", Smax)




