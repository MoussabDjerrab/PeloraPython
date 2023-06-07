import numpy as np 
#step 1 : ***********************************************************************************************
p= 6
n= 8
X = np.random.randn(p, n)
print("la matrice X")
print(X)
Y = np.random.choice([0, 1], size=n)
print("le vecteur Y")
print(Y)

#step 2 : ************************************************************************************************
#calcule des scores
def calc_scor(N0V,N1V):
    score = 0
    for j in range(0,len(N0V)):
        for k in range(0,len(N1V)):
            if N0V[j]>=N1V[k]:
                score = score+1 
    return score 

scores = np.empty(shape=(0,))
for i in range(0,p):
    tab = X[i]
    N0V = tab[Y==0]
    N1V = tab[Y==1]
    score = calc_scor(N0V,N1V)
    scores = np.append(scores,score)

print("vecteur de scores ")
print(scores)

#calcule de Smax 
n0 = len(Y) - np.count_nonzero(Y)
n1 = np.count_nonzero(Y)
Smax = n0*n1
print("smax")
print(Smax)

#la nouvelle matrice en comparant le score avec smax  
Xprime = X 
for i in range(0,len(scores)):
    if scores[i]>Smax/2:
        Xprime[i] = -1*Xprime[i]

print("X' : ")
print(Xprime)

#le nouveau scores 
nv_scores = np.empty(shape=(0,))
for i in range(0,len(scores)):
    scor = min (scores[i],Smax-scores[i])
    nv_scores = np.append(nv_scores,scor)

print("new scores ")
print(nv_scores)

min_nv_scores = min(nv_scores)
print("min valeur de score")
print(min_nv_scores)

#step 3 : ******************************************************************************************************
#if the initiale cluster is not given
#chercher i*
indices = np.empty(shape=(0,))
sumi = 0
for i in range(0,len(nv_scores)):
    if nv_scores[i]==min_nv_scores :
        indices =np.append(indices,i)
        sumi = sumi+1

marges = np.empty(shape=(0,))
if sumi==1 : 
    i_prim = indices[0]
else :#calcule des margins
    for i in range(0,len(indices)):
        VEC=Xprime[int(indices[i])]
        N0V = VEC[Y==0]
        N1V = VEC[Y==1]
        marg = min(N1V) - max(N0V)
        marges = np.append(marges,marg)

    indice_max_marge = np.argmax(marges)  
    i_prim = indices[indice_max_marge]    
print("i* : ")
print(i_prim)

#le cluster initial 
initial_cluster = np.empty(shape=(0,))
initial_cluster = Xprime[int(i_prim)]

#step 3 : initial cluster is given *****************************************************************************
nbr_cluster = 2 

def decouper_matrice(matrice, n):
    m, _ = matrice.shape  # Récupère le nombre de lignes de la matrice d'origine
    matrices_decoupees = np.split(matrice, n)
    return matrices_decoupees

clusters = decouper_matrice(Xprime,nbr_cluster)    
initial_cluster_given = clusters[0] 

print("initial cluster given ")
print(initial_cluster_given)
moyenne_genes_cluster = np.mean(initial_cluster_given, axis=0)
len_C =len(initial_cluster_given) 


#Step 4 and 5: ******************************************************************************************************
#score initiale of the initial cluster given 
N0V = moyenne_genes_cluster[Y==0]
N1V = moyenne_genes_cluster[Y==1]
score_initial = calc_scor(N0V,N1V) 

def add_to_cluster(cluster_p,indices=[]):
    indice= np.empty(shape=(0,))
    scores_avg = np.empty(shape=(0,))
    avg = np.empty(shape=(0,))
    for i in range(len_C,p):
        if i not in indices : 
            indice = np.append(indice,i)
            cluster_p = np.vstack((cluster_p,Xprime[i]))
            avg = np.mean(cluster_p, axis=0)
            N0V= avg[Y==0]
            N1V = avg[Y==1]
            score= calc_scor(N0V,N1V)  
            scores_avg = np.append(scores_avg,score)
            cluster_p = cluster_p[:-1, :]
   
    if np.size(scores_avg)!= 0: 
        min_score = min(scores_avg)
        if min_score < score_initial : 
            ind_min = np.argmin(scores_avg)
            gene = Xprime[int(indice[ind_min])]
            indices = np.append(indices,int(indice[ind_min]))
            cluster_p = np.vstack((cluster_p,gene))
            add_to_cluster(cluster_p,indices)
        else :
            print(cluster_p)
            return cluster_p 
    else :
        print(cluster_p)
        return cluster_p 

print("le cluster obtenu")
cluster_nv = add_to_cluster(initial_cluster_given)
