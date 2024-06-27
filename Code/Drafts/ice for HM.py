# Algorithme ICE dans le cas gaussien 

'''
Attention les notations sont inversées par rapport aux articles de W. PIECZYNSKI.
Nous nous sommes basé sur les notations de M. CASTELLA.
Notamment R=(r1,...,rn) représente ici les états cachés tandis que X=(x1,...,xn) correspond aux observations. 
'''
import numpy as np
import random as rd

def generer_donnees_Markov(n,mu,sigma,pi, P_hat) :
    ''' Fonction de génération des données dans le cas où les sources peuvent être modélisées par une chaîne de Markov cachée
    Parametres
    ----------
    n = taille de l'échantillon
    mu = matrice 1x2 composée de mu0 et mu1 les moyennes des lois conditionnelles à x=0 et x=1 resp.
    sigma = matrice 1x2 composée de sig0 et sig1 les ecarts-types des lois conditionnelles à x=0 et x=1 resp.
    pi = proba que r0=0
    P_hat = matrice 2x2 des probabilités de transition entre les états cachés


    Return
    ------
    r = matrice de taille n des états cachés
    x = matrice de taille n des observations
    '''
    r = np.random.rand(n)
    r[0] = r[0] > pi
    x = np.zeros(n)
    x[0] = rd.normalvariate(mu[int(r[0])],sigma[int(r[0])])
    for i in range(1,n) : #on met r à 0 ou à 1 en fonction de P et on tire les observations
        r[i] = r[i] > P_hat[0][int(r[i-1])]
        x[i] = rd.normalvariate(mu[int(r[i])],sigma[int(r[i])])
    return r, x
        
'''def tirage(n,pi_hat,P_hat)  :
    Fonction tirant des sets de X, utilisée dans les itérations de l'algo ICE pour estimer mu et sigma
    Parametres
    ----------
    n = taille de l'échantillon
    prx = matrice de taille n indiquant les probas ri=0 sachant x
    mu = matrice 2x1 composée de mu0 et mu1 les moyennes des lois conditionnelles à x=0 et x=1 resp.
    sigma = matrice 2x1 composée de sig0 et sig1 les ecarts-types des lois conditionnelles à x=0 et x=1 resp.

    Return
    ------
    r = matrice de taille n des états cachés supposés

    
    r = np.random.rand(n)
    r[0] = r[0] > pi_hat
    for i in range(1,n) : #on met r à 0 ou à 1 en fonction de gamma et psi
        val = int(r[i-1])
        r[i] = r[i] > P_hat[0][val]
    return r'''

def tirage(n,pi_hat,gamma)  :
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
    
    r = np.random.rand(n) > gamma
    return r


def forward_backard(P_hat,pi_hat,Pxr,nEtats,n):
    #Initialisation
    alpha = np.zeros((nEtats, n))
    beta = np.zeros((nEtats, n))
    
    #Partie Foward
    alpha[0][0] = pi_hat * Pxr[0][0]
    alpha[1][0] = (1-pi_hat) * Pxr[1][0]
    alpha[:,0] = alpha[:,0]/np.sum(alpha[:,0])
    
    
    for i in range(1,n):
        alpha[:,i] = (alpha[:,i-1] * np.diag(P_hat) * Pxr[:,i])
        #coeffrenorm = Pxr[0][i]*alpha[0,i-1]*P_hat[0][0] + Pxr[1,i]*alpha[0][i-1]*P_hat[1][0] + Pxr[0][i]*alpha[1,i-1]*P_hat[0][1] + Pxr[1,i]*alpha[1][i-1]*P_hat[1][1]
        #alpha[:,i] = alpha[:,i]/coeffrenorm
   
    print(f"alpha{alpha}")

    #Partie Backward    
    beta[:,n-1]=[1,1]
    
    for i in range(n-1,0,-1):
        beta[0,i-1] = P_hat[0][0]*Pxr[0][i]*beta[0,i] + P_hat[1][0]*Pxr[1][i]*beta[0,i]
        beta[1,i-1] = P_hat[0][1]*Pxr[0][i-1]*beta[1,i] + P_hat[1][1]*Pxr[1][i-1]*beta[1,i]
        #coeffrenorm = Pxr[0][i]*alpha[0,i-1]*P_hat[0][0] + Pxr[1,i]*alpha[0][i-1]*P_hat[1][0] + Pxr[0][i]*alpha[1,i-1]*P_hat[0][1] + Pxr[1,i]*alpha[1][i-1]*P_hat[1][1]
        #print(f"coeff{coeffrenorm}")
        #beta[:,i-1] = beta[:,i-1]/coeffrenorm

    print(f"beta{beta}")
   
    gamma = np.zeros((nEtats, n))

    gamma[0] = alpha[0]*beta[0]
    gamma[1] = alpha[1]*beta[1]
    
    psi = np.zeros((nEtats, nEtats, n-1))

    for i in range(0,n-1):
        psi[:,:,i] = (alpha[:,i]* (Pxr[:,i+1] * beta[:,i+1]) ) * P_hat
        coeffrenorm = np.sum(np.sum( psi[:,:,i]))
        psi[:,:,i] = psi[:,:,i] / coeffrenorm

    return gamma,psi

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
    '''
    #initialisation
    pi_hat = 0.5 #proba que R0=0
    R_tirage = np.zeros((N,n)) # représente les N tirages qui seront fait à chaque itération
    R_hat = np.random.binomial(1,0.5,n) #On tire un premier set d'états cachés

    U1 = R_hat.sum() #U0 et les V sont calculés comme définis dans notre formalisation du pbm
    V0 = ((1-R_hat) * X).sum()
    V1 = ((R_hat) * X).sum() 

    mu_hat = np.array([V0/(n-U1),V1/U1])

    S0=0
    S1=0
    S0 = np.sum((1-R_hat)*(X - mu_hat[0]) ** 2)
    S1 = np.sum((R_hat)*(X - mu_hat[1]) ** 2)

    sigma_hat = np.array([np.sqrt(S0/(n-U1)),np.sqrt(S1/U1)])

    Pxr = np.empty((2, n)) #proba de x sachant r
    P_hat = (np.eye(2) * 0.5) +(np.ones((2, 2))- np.eye(2)) * 1/(2*(2-1))
    
    sig0_hat_tirage = np.zeros(N)
    sig1_hat_tirage = np.zeros(N)
    
    for _ in range (q):
        Pxr[0,:]=(1/(2*np.pi*sigma_hat[0]**2))*np.exp(-(X-mu_hat[0])**2/(2*sigma_hat[0]**2))
        Pxr[1,:]=(1/(2*np.pi*sigma_hat[1]**2))*np.exp(-(X-mu_hat[1])**2/(2*sigma_hat[1]**2))
        print(f"Pxr:{Pxr}")     
        gamma, psi = forward_backard(P_hat,pi_hat,Pxr,2,n)
        print(f"gamma{gamma}")
        print(f"psi{psi}")
        P_hat[0][0] = np.sum(psi[0,0,:])/np.sum(gamma[0])
        P_hat[0][1] = np.sum(psi[0,1,:])/np.sum(gamma[0])
        P_hat[1][0] = 1 - P_hat[0][0]
        P_hat[1][1] = 1 - P_hat[0][1]

        print(f"P_hat{P_hat}")
        pi_hat = gamma[0][0]

        
        #on effectue N tirage de R
        for j in range(N) :
            R_tirage[j]= tirage(n,pi_hat,gamma[0,:])

        #ce qui suit est très moche mais on optimisera plus tard...
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
            if U0==0 :
                U0=1
            mu0_hat_tirage[j] = V0/U0
            if U0==n :
                U0=0
            mu1_hat_tirage[j] = V1/(n-U0)

        mu_hat[0] = mu0_hat_tirage.sum()/N
        mu_hat[1] = mu1_hat_tirage.sum()/N
        print(f"mu:{mu_hat}")
    
        S0 = 0
        S1 = 0
        #calcul de sigma
        for j in range(N):
            U0 = np.sum(1-R_tirage[j])
            S0 = np.sum((1-R_tirage[j])*(X - mu_hat[0]) ** 2)
            S1 = np.sum((R_tirage[j])*(X - mu_hat[1]) ** 2)
            if U0==0 :
                sig1_hat_tirage[j] = np.sqrt(S1/(n-U0))
            elif U0==n :
                sig0_hat_tirage[j] = np.sqrt(S0/U0)
            else : 
                sig0_hat_tirage[j] = np.sqrt(S0/U0)
                sig1_hat_tirage[j] = np.sqrt(S1/(n-U0))
        sigma_hat[0] = sig0_hat_tirage.sum()/N
        sigma_hat[1] = sig1_hat_tirage.sum()/N
        print(f"sigma:{sigma_hat}")

    return mu_hat,sigma_hat, P_hat, tirage(n,pi_hat,P_hat)


if __name__ == '__main__':
    
    #paramètres du modèles
    mu = [4,7]
    sigma = [0.2,0.2]
    pi = 0.7
    P = np.array([[0.7,0.5],[0.3,0.5]])

    #Generation des datas
    n = 15 #nombres d'observations
    r,x = generer_donnees_Markov(n,mu,sigma,pi, P)
  
    #Utilisation de l'algo ICE
    q=4 #nombre d'itérations de l'algo
    N=10 #nombre de tirages effectués à chaque itération
    mu_hat,sigma_hat, P_hat, r_hat = ice(n,x,q,N)
    
    print("L'algo ICE renvoie comme paramètres")
    print(f"P : {P_hat}")
    print(f"mu : {mu_hat}")
    print(f"sigma : {sigma_hat}")
    print(f"Le taux de correspondance entre la séquence retrouvée et celle réellement émise est : {(r_hat== r).sum()/n}")