import numpy as np

A=np.array([1,2,3,4])
print(A.size)
print(A.ndim)
print(A.shape)

B=np.zeros((10,2)) #mettre la shape en tuple
print(B)

C=np.ones((14,2))
print(C)
print(C.size) #produit entre les dimensions

D=np.random.randn(3,4) #nombres aléatoires selon normale
print(D)

E=np.eye(4,4)
print(E) #identité 

F=np.linspace(0,10,20) #début,fin,nombre
print(F)
G=np.arange(0,10,1) #début,fin,pas
print(G)

########Concaténation#######

A=np.zeros((3,2))
print(A)
B=np.ones((3,2))
print(B)

#Concaténation horizontale
C=np.hstack((A,B))
print(C)

#Concaténation verticale
D=np.vstack((A,B))
print(D)

#autre méthode à retenir impérativement
C=np.concatenate((A,B),axis=0) #0 vertical
print(C)
D=np.concatenate((A,B),axis=1) #0 horizontal
print(D)

#reshape !
print(C.shape)
C=np.reshape((3,4))
print(E)

#en pratique 
A=np.array([1,2,3,4])
print(A.shape) #il manque le 1
A=A.reshape((A.shape[0],1))
print(A.shape) #le 1 est dispo

A=A.ravel()

def init(m,n):
    biais=np.ones((m,1))
    M=np.random.randn(m,n)
    retour=np.concatenate((M,biais),axis=1)
    return retour 

A=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(A)
print(A[1,0])

#Slicing 
print(A[:,-2:]) #pour sélectionner les 2 dernières colonnes
A=np.zeros((4,4))
print(A)
A[1:3,1:3]=1

B=np.zeros((5,5))
B[::2,::2]=1
print(B)

#Boolean indexing 
A=np.random.randint(0,10,[5,5])
print(A<5) #tous les éléments inf. à 5 de A
#il s'agit d'un mask 
A[A<4]=0
print(A)

#exercice zoom image sclicing 
from skimage import data, color
import matplotlib.pyplot as plt

face = data.astronaut()  # Image en couleur
face_gray = color.rgb2gray(face)  # Conversion en niveaux de gris

h=face_gray.shape[0]
L=face_gray.shape[1]
print(h,L)
image=face_gray[h//4:-h//4,L//4:-L//4]
image[image>150]=255
plt.imshow(image, cmap="gray")  # Affichage en noir et blanc
plt.axis("on")
plt.show()

###Méthodes usuelles
A=np.random.randint(0,10,[5,5])
print(A.sum())
print(A.sum(axis=0))#colonnes 
print(A.sum(axis=1))#lignes
print(A.min(axis=0))
print(A.min(axis=1))
print(A.argmin(axis=0)) #index du min sur l'axe 0

print(np.exp(A))


###Statistiques
print(A.mean(axis=0))
print(A.std(axis=0))

print(np.corrcoef(A)) #corrélation entre les lignes
#comptage des différents éléments dans le tableau
print(np.unique(A,return_counts=True)) 
values, count=np.unique(A,return_counts=True)
print(count)
print(A)

for i, j in zip(values[count.argsort()],count[count.argsort()]):
    print(f'valeur {i} apparaît {j} fois')

#####valeurs NaN########
A=np.random.randn(5,5)
print(A)
A[0,2]=np.nan
A[3,4]=np.nan
print(np.nanmean(A)) #ingnore les NaN

print(np.isnan(A).sum()/A.size) #compter les NaN puis faire le rapport 
A[np.isnan(A)]=0 #remplace par des 0 
print(A)

###Exercice###
np.random.seed(0)
A=np.random.randint(0,100,[10,5])
print(A)
moy=np.mean(A,axis=0)
ecart=np.std(A,axis=0)
print(moy)
c=A.shape[1]
print((A-moy)/ecart)

##broadcasting 
#méthode reshape primordiale 
b=np.random.randint(0,5,[2,2])
c=np.random.randint(0,5,[2,2])
print(b)
print(c)
print(b+c)


