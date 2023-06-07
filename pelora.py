import numpy as np

def generate_matrix(n=100, p=10):
    rg = np.random.default_rng(1)
    x = rg.random((n, p))
    y = np.random.randint(2, size=n)

    return x, y

def get_gene_scores(x, y):
    class_0 = []
    class_1 = []

    for i in range(y.size):
        if y[i] == 0:
            class_0.append(x[i])
        else:
            class_1.append(x[i])

    class_0 = np.array(class_0)
    class_1 = np.array(class_1)

    nb_genes = int(x.size / y.size)

    gene_scores = np.zeros(nb_genes)

    for i in range(nb_genes):
        for j in class_0:
            for l in class_1:
                if j[i] >= l[i]:
                    gene_scores[i] += 1



    return gene_scores

def get_average(a1, a2):
    cluster = 0.5 * (a1 + a2)
    return cluster


def forward_search(x, y, initial_cluster_mean):

    class_0_ind = []
    class_1_ind = []

    for i in range(y.size):
        if y[i] == 0:
            class_0_ind.append(i)
        else:
            class_1_ind.append(i)

    class_0_ind = np.array(class_0_ind)
    class_1_ind = np.array(class_1_ind)

    nb_genes = int(x.size / y.size)

    lowest_score = 10000000
    lowest_score_index = 0


    for i in range(nb_genes):
        gene_expression = get_average(initial_cluster_mean, x[:, i])
        gene_score = 0
        for j in class_0_ind:
            for l in class_1_ind:
                if gene_expression[j] >= gene_expression[l]:
                    gene_score += 1
        if gene_score < lowest_score:
            lowest_score = gene_score
            lowest_score_index = i

    return lowest_score_index

def merge(x, g1_index, g2_index):
    # Remove columns 0 and 2
    updated_arr = np.delete(x, [g1_index, g2_index], axis=1)

    # Merge columns 0 and 2 by adding their values together
    merged_column = get_average(x[:, g1_index], x[:, g2_index])

    # Add the merged column to the updated array
    result = np.column_stack((updated_arr, merged_column))

    return result



def pelora():
    # Initializing
    x, y = generate_matrix(10, 3)

    # calculate gene scores
    scores = get_gene_scores(x, y)

    # index of gene with lowest score
    min_index = np.argmin(scores)

    # get cluster mean
    initial_cluster_mean = x[:, min_index]

    i = forward_search(x, y, initial_cluster_mean)
    print(i)



