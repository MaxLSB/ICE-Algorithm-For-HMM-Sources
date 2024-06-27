# Algorithme ICE dans le cas gaussien 

'''
Attention les notations sont inversées par rapport aux articles de W. PIECZYNSKI.
Nous nous sommes basé sur les notations de M. CASTELLA.
Notamment R=(r1,...,rn) représente ici les états cachés tandis que X=(x1,...,xn) correspond aux observations. 
'''
import numpy as np
import random as rd

def generer_donnees(n,mu,sigma,alpha) :
    ''' Fonction de génération des données
    Parametres
    ----------
    n = taille de l'échantillon
    mu = matrice 1x2 composée de mu0 et mu1 les moyennes des lois conditionnelles à x=0 et x=1 resp.
    sigma = matrice 1x2 composée de sig0 et sig1 les ecarts-types des lois conditionnelles à x=0 et x=1 resp.
    alpha = proba que ri=0

    Return
    ------
    r = matrice de taille n des états cachés
    x = matrice de taille n des observations
    '''
    r = np.random.rand(n) > alpha

    x = np.zeros(n)
    for i in range(n) : #on tire les observations en fonction de r
        source = int(r[i])
        x[i] = rd.normalvariate(mu[source],sigma[source])
    return r, x
        
def tirage(n,prx,mu,sigma):
    ''' Fonction tirant des sets de X, utilisée dans les itérations de l'algo ICE pour estimer mu et sigma
    Parametres
    ----------
    n = taille de l'échantillon
    prx = matrice de taille n indiquant les probas ri=0 sachant x
    mu = matrice 2x1 composée de mu0 et mu1 les moyennes des lois conditionnelles à x=0 et x=1 resp.
    sigma = matrice 2x1 composée de sig0 et sig1 les ecarts-types des lois conditionnelles à x=0 et x=1 resp.

    Return
    ------
    r = matrice de taille n des états cachés supposés
    '''
    r = np.random.rand(n) > prx
    return r

def ice(n,X,q,N):
    ''' Fonction de classification itérative
    Parametres
    ----------
    n = taille de l'échantillon
    X = matrice de taille n des observations
    q = nombres d'itérations globales de l'algo ICE
    N = nombre de tirages de X à faire à chaque itération
    
    Return
    ------
    mu = matrice 2x1 composée de mu0 et mu1 les moyennes des lois conditionnelles à x=0 et x=1 resp.
    sigma = matrice 2x1 composée de sig0 et sig1 les ecarts-types des lois conditionnelles à x=0 et x=1 resp.
    séquence des états cachés supposés
    '''
    #initialisation
    R_tirage = np.zeros((N,n)) # représente les N tirages qui seront fait à chaque itération
    R_hat = np.random.binomial(1,0.5,n) #On tire un premier set d'états cachés

    U1 = R_hat.sum() #U0 et les V sont calculés comme définis dans notre formalisation du pbm
    V0 = ((1-R_hat) * X).sum()
    V1 = ((R_hat) * X).sum() 

    alpha_hat = 0.5 #proba que Ri=0
    mu_hat = np.array([V0/(n-U1),V1/U1])

    S0=0
    S1=0
    S0 = np.sum((1-R_hat)*(X - mu_hat[0]) ** 2)
    S1 = np.sum((R_hat)*(X - mu_hat[1]) ** 2)

    sigma_hat = np.array([np.sqrt(S0/(n-U1)),np.sqrt(S1/U1)])

    Pxr = np.empty((2, n)) #proba de x sachant r
    Px_inter_r = np.empty((2, n)) #proba de x inter r
    Prx = np.empty((n,)) #proba de x sachant r
    
    for _ in range (q):
        #On évalue les proba de xi sachant ri=0 et sachant ri=1
        Pxr[0,:]=(1/(2*np.pi*sigma_hat[0]**2))*np.exp(-(X-mu_hat[0])**2/(2*sigma_hat[0]**2))
        Pxr[1,:]=(1/(2*np.pi*sigma_hat[1]**2))*np.exp(-(X-mu_hat[1])**2/(2*sigma_hat[1]**2))
   
        #On évalue les proba de (xi inter ri=0) et (xi inter ri=1)
        Px_inter_r [0,:] = Pxr[0,:] * alpha_hat
        Px_inter_r [1,:] = Pxr[1,:] * alpha_hat

        #On évalue la proba que ri=0 sachant xi 
        Prx = Px_inter_r[0,:]/(Px_inter_r[0,:]+Px_inter_r[1,:])
        
        #ré-estimation alpha
        alpha_hat = (Prx.sum())/n

        #on effectue N tirage de R
        for j in range(N) :
            R_tirage[j]= tirage(n,Prx,mu_hat,sigma_hat) 
        

        #ré-estimation de mu et sigma par tirage de N sets de R et calcul empirique
        U0=0
        V0=0
        V1=0
        #calcul de mu
        mu0_hat_tirage = np.zeros(N)
        mu1_hat_tirage = np.zeros(N)
        for j in range(N) :
            U0 = np.sum(1-R_tirage[j])
            V0 = np.sum((1-R_tirage[j])*X)
            V1 = np.sum(R_tirage[j]*X)
            mu0_hat_tirage[j] = V0/U0
            mu1_hat_tirage[j] = V1/(n-U0)

        mu_hat[0] = mu0_hat_tirage.sum()/N
        mu_hat[1] = mu1_hat_tirage.sum()/N
        
        S0 = 0
        S1 = 0
        #calcul de sigma
        sig0_hat_tirage = np.zeros(N)
        sig1_hat_tirage = np.zeros(N)
        for j in range(N):
            U0 = np.sum(1-R_tirage[j])
            S0 = np.sum((1-R_tirage[j])*(X - mu_hat[0]) ** 2)
            S1 = np.sum((R_tirage[j])*(X - mu_hat[1]) ** 2)
            sig0_hat_tirage[j] = np.sqrt(S0/(U0))
            sig1_hat_tirage[j] = np.sqrt(S1/(n-U0))
       

        sigma_hat[0] = sig0_hat_tirage.sum()/N
        sigma_hat[1] = sig1_hat_tirage.sum()/N

    return mu_hat,sigma_hat, alpha_hat, tirage(n,Prx,mu_hat, sigma_hat)


if __name__ == '__main__':
    
    #paramètres du modèles
    mu = [4,7]
    sigma = [0.7,0.7]
    alpha = 0.7

    #Generation des datas
    n = 20 #nombres d'observations
    r,x = generer_donnees(n,mu,sigma,alpha)
  
    #Utilisation de l'algo ICE
    q=200 #nombre d'itérations de l'algo
    N=10 #nombre de tirages effectués à chaque itération
    mu_hat,sigma_hat, alpha_hat, r_hat = ice(n,x,q,N)
    
    print("L'algo ICE renvoie comme paramètres")
    print(f"alpha : {alpha_hat}")
    print(f"mu : {mu_hat}")
    print(f"sigma : {sigma_hat}")
    print(f"Le taux de correspondance entre la séquence retrouvée et celle réellement émise est : {(r_hat== r).sum()/n}")