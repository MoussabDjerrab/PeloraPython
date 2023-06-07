import numpy as np 

#************************************************************** Etape 1 : -------------------------------------------------------------
#generation de la matrice des genes :
p= 6
n= 8
X = np.random.randn(p, n)
print("la matrice X")
print(X)
Y = np.random.choice([0, 1], size=n)
print("le vecteur Y")
print(Y)

#************************************************************* Etape 2 : ---------------------------------------------------------
#boucle pour le calcule de score : 
scores = np.empty(shape=(0,))
for i in range(0,p):
    score = 0
    N0 = np.empty(shape=(0,))
    N1 = np.empty(shape=(0,))
    for m in range(0,n):
        if Y[m]==0 :
           N0 =np.append(N0,X[i,m])
        else : 
            if Y[m]==1 : 
                N1=np.append(N1,X[i,m]) 
    for j in range(0,len(N0)):
        for l in range(0,len(N1)):
            if N0[j]>=N1[l]:
                score = score+1 
  
    scores = np.append(scores,score)

print("vecteur de scores ")
print(scores)

#----------- calcule de "Smax  = NO * N1 " :--------------------- 
n0 = len(Y) - np.count_nonzero(Y)
n1 = np.count_nonzero(Y)
Smax = n0*n1
print("smax est :")
print(Smax)

#la matrice d'expression elle est déduite en comparant les  scores avec le Smax/2 en multipliant par (-1):

expression = np.random.randn(p, n)
expression = X 

for i in range(0,len(scores)):
    if scores[i]>Smax/2:
       expression[i] *= -1

print(" l'expression : ")
print(expression)


#----------------------- nouveaux scorex  en choisissant le min entre "score initial  et Smax-score" :------------
new_score = np.empty(shape=(0,))
for i in range(0,len(scores)):
    scor = min (scores[i],Smax-scores[i])
    new_score  = np.append(new_score ,scor)

print("le nouveau score est : ")
print(new_score )

#-**************************************************************Etape 3 : a- *****************************************************************************

#-------------------------------------------- cherchons les min des scores entre les nouveau score --------------------------
min_new_score  = min(new_score )
print("valeur min des scores est :")
print(min_new_score )

#------------------------------------------------------------------dans le cas ou le clustor initiale est donnée : ----------------------------
#------------------------------------------------- cherchons le gene star (i*):------------------------------------------
ind = np.empty(shape=(0,))
somme_i = 0
for i in range(0,len(new_score )):
    if new_score [i]==min_new_score  :
        ind =np.append(ind,i)
        somme_i += 1
#-----------------------------calculons la marge pour identifier le i star dans le cas ou y'a plusieur gene avec le mm score  ----------
marges = np.empty(shape=(0,))
if somme_i==1 : 
    i_starr = ind[0]
else :              
    for i in range(0,len(ind)):
        N0 = np.empty(shape=(0,))
        N1 = np.empty(shape=(0,))
        for m in range(0,n):
            if Y[m]==0 :
                ind = int(ind[i])
                
                N0 =np.append(N0,expression[ind,m])
            else : 
                if Y[m]==1 : 
                    ind = int(ind[i])
                    N1=np.append(N1,expression[ind,m]) 
        
        merge = min(N1) - max(N0)
        marges = np.append(marges,merge)

    marge_max = np.argmax(marges)  
    i_starr = ind[ marge_max]    
print(" gene star 'i*' est : ")
print(i_starr)

#-------------------------------------------------- identification de clustor initial : --------------------------------------
cluster_initial = np.empty(shape=(0,))
cluster_initial  = expression[int(i_starr)]

#------------------------------------------------ Etape 3 : b- le clustor initial est donnée  :-----------------------
nombre_clus = 2 

def decouper_matrice(matrice, n):
    m, _ = matrice.shape                         # Récupère le nombre de lignes de la matrice d'origine
    matrices_decoupees = np.split(matrice, n)
    return matrices_decoupees

clust = decouper_matrice(expression,nombre_clus)    
cluster_donnée = clust[0] 

print(cluster_donnée)
avg_cluster = np.mean(cluster_donnée, axis=0)
print(avg_cluster)


