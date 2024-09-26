import numpy as np
import random as rd

def generer_donnees_Markov(n,mu,sigma,pi,P_hat) :
    
    r = np.random.rand(n)
    r[0] = r[0] > pi
    x = np.zeros(n)
    x[0] = rd.normalvariate(mu[int(r[0])],sigma[int(r[0])])
    for i in range(1,n) : #on met r à 0 ou à 1 en fonction de P et on tire les observations
        r[i] = r[i] > P_hat[0][int(r[i-1])]
        x[i] = rd.normalvariate(mu[int(r[i])],sigma[int(r[i])])
    return r, x


def forward_backward(P_hat,pi_hat,Pxr,nEtats,n):
    #Initialisation  
    alpha = np.zeros((nEtats, n))
    beta = np.zeros((nEtats, n))

    #Partie Foward
    alpha[:, 0] = pi_hat * Pxr[:, 0]
    alpha[:, 0] = alpha[:, 0] / np.sum(alpha[:, 0])
    for t in range(1, n):
        alpha[:, t] = (alpha[:, t-1] @ P_hat) * Pxr[:, t]
        alpha[:, t] = alpha[:, t] / np.sum(alpha[:, t])

    #Partie Backward
    beta[:, n-1] = [1, 1]
    for t in range(n-1, 0, -1):
        beta[:, t-1] = P_hat @ (beta[:, t] * Pxr[:, t])
        coeffrenorm = alpha[:, t-1].T @ P_hat @ Pxr[:, t]
        beta[:, t-1] = beta[:, t-1] / coeffrenorm

    # Autres coefficients
    gamma = alpha * beta  #
    
    psi = np.zeros((nEtats, nEtats, n-1))
    for t in range(n-1):
        psi[:, :, t] = (alpha[:, t].reshape(-1, 1) * (Pxr[:, t+1] * beta[:, t+1])) * P_hat
        coeffrenorm = np.sum((alpha[:, t].reshape(-1, 1) * (Pxr[:, t+1] * beta[:, t+1])) * P_hat)
        psi[:, :, t] = psi[:, :, t] / coeffrenorm

    return gamma, psi

def ice(n,x,q,N,nEtats=2):
    ''' Algorithme ICE pour la segmentation de séquences
    '''

    pi_hat_Ini = np.ones(nEtats) / nEtats
    P_hat_Ini = 0.5 * np.eye(nEtats) + (np.ones((nEtats, nEtats)) - np.eye(nEtats)) * 1 / (2 * (nEtats - 1))

    mu_Ini = np.zeros(nEtats)

    if nEtats % 2 == 0:
        for i in range((nEtats + 1) // 2):
            mu_Ini[i] = np.mean(x) - (((nEtats // 2) - (i)) * (np.std(x) / 2))
            mu_Ini[nEtats - (i + 1)] = np.mean(x) + (((nEtats // 2) - (i)) * (np.std(x) / 2))
    else:
        for i in range(nEtats):
            mu_Ini[i] = np.mean(x) - (((nEtats // 2) - (i)) * (np.std(x) / 2))

    sigma_Ini = np.std(x)

    mu_hat = mu_Ini
    sigma_hat = np.tile(sigma_Ini, nEtats)
    pi_hat = pi_hat_Ini
    P_hat = P_hat_Ini

    for ii in range(q):
        # Calcul de la proba conditionnelle à état
        Pxr = np.zeros((nEtats, n))
        Pxr[0, :] = 1 / (sigma_hat[0] * np.sqrt(2 * np.pi)) * np.exp(- (x - mu_hat[0]) ** 2 / (2 * sigma_hat[0] ** 2))
        Pxr[1, :] = 1 / (sigma_hat[1] * np.sqrt(2 * np.pi)) * np.exp(- (x - mu_hat[1]) ** 2 / (2 * sigma_hat[1] ** 2))

        #Appel de la fonction forward_backward
        gamma, psi = forward_backward(P_hat,pi_hat,Pxr,nEtats,n)

        # Réestimation des paramètres
        pi_hat = 1 / n * np.sum(gamma, axis=1)
        P_hat = np.sum(psi, axis=2) / np.sum(gamma[:, :n-1], axis=1).reshape(-1, 1)

        # Simulation des variables cachées selon la loi a posteriori
        r_ice = np.zeros((N, n), dtype=int)
        mutmp = np.zeros((N, nEtats))
        sigmatmp = np.zeros((N, nEtats))
        for n_ice in range(N):
            r_ice[n_ice, 0] = np.random.rand() > gamma[0, 0]
            for t in range(n-1):
                Pposteriorihat = psi[:, :, t] / gamma[:, t].reshape(-1, 1)
                CPposteriorihat = np.cumsum(Pposteriorihat, axis=1)
                U = np.random.rand()
                r_ice[n_ice, t+1] = np.sum(U > CPposteriorihat[r_ice[n_ice, t], :])
            mutmp[n_ice, 0] = np.mean(x[np.where(r_ice[n_ice, :] == 0)])
            mutmp[n_ice, 1] = np.mean(x[np.where(r_ice[n_ice, :] == 1)])
            sigmatmp[n_ice, 0] = np.std(x[np.where(r_ice[n_ice, :] == 0)])
            sigmatmp[n_ice, 1] = np.std(x[np.where(r_ice[n_ice, :] == 1)])

        mu_hat = np.mean(mutmp, axis=0)
        sigma_hat = np.mean(sigmatmp, axis=0)


    return mu_hat, sigma_hat, P_hat, gamma

if __name__ == '__main__':
    
    #paramètres du modèles
    mu = [0,2]
    sigma = [1,1]
    pi = 0.5
    P_hat = np.array([[0.7, 0.3], [0.7, 0.3]])

    #Generation des datas
    n = 500 #nombres d'observations
    r,x = generer_donnees_Markov(n,mu,sigma,pi, P_hat)
  
    #Utilisation de l'algo ICE
    q=500 #nombre d'itérations de l'algo
    N=1 #nombre de tirages effectués à chaque itération
    mu_hat,sigma_hat, P_hat, gamma = ice(n,x,q,N,nEtats=2)

    r_segm = np.argmax(gamma, axis=0)
    r_segm = r_segm - 1
    
    print("L'algo ICE renvoie comme paramètres")
    print(f"P : {P_hat}")
    print(f"mu : {mu_hat}")
    print(f"sigma : {sigma_hat}")
    print('Taux bonne segmentation: ', np.sum(r == r_segm) / n)



