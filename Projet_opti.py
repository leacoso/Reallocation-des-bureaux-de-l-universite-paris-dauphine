import cvxpy as cvx
import numpy as np
from numpy.random import multivariate_normal, randn, uniform, choice
from scipy.linalg import norm
from scipy.linalg import solve
from math import sqrt
import time
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt


#QUESTION 1 
p = 5  # nombre de phases
n = 5  # nombre d'ailes

#récupère l'indice du vecteur qui correspond au déplacement des bureaux de l'aile i à l'aile j lors de la phase p
def index(i,j,k):
    return (i - 1) * n * (p + 1) + (j - 1) * (p + 1) + k

def index_Y(i,j,k,l):
    return (i-1) * (n-1) * n * (p+1) + (j-1) * n * (p+1) + k * n + l-1

def index_P(i,j,k):
    return   (i - 1) * (n-1) * (p + 1) + (j - 1) * (p + 1) + k

#créé le vecteur tel qu'il y a 1 lorsque les bureaux se déplacent et 0 pour toutes les fois où les bureaux restent sur place
def give_equality():
    I = np.ones(n*n*(p+1))

    for i in range(n):
        for k in range(p+1):
            ind = index(i+1,i+1,k)
            I[ind] = 0
    return I

def modele_lineaire_5_bureaux():

    x = cvx.Variable((n*n*(p+1)))
    I = give_equality()
    P = cvx.Variable((n*(n-1)*(p+1)))
    Y = cvx.Variable((n*n*n*(p+1)))

    function = cvx.Minimize(x.T @ I)
    contraintes = []

    """Contraintes 1 : Initialisation des positions"""
    #La configuration est la même au debut est à l'arrivée
    for i in range(1,n):
        contraintes.append(P[index_P(i,i,0)] == 1)
        contraintes.append(P[index_P(i,i,5)] == 1)

    #A la phase 0 personne ne bouge
    contraintes.append(sum( sum( x[index(i,j,0)] * I[index(i, j, 0)]  for j in range(1,n+1) ) for i in range(1,n+1)) == 0)

    """Contrainte 2 :

    Pour chaque phase, il y a un 'mouvement' dans chaque aile """

    for k in range(1,p+1):
        if (k == 1):
            for i in range(1,n) :
                contraintes.append( sum( x[index(i,j,k)] for j in range(1,n+1)) == 1)
            contraintes.append(sum( x[index(n,j,k)] for j in range(1,n+1)) == 0)
        else :
            for i in range(1,n+1):
                contraintes.append( sum( x[index(i,j,k)] for j in range(1,n+1)) == 1)

    """Contrainte 3 :
    Contraintes linéaire pour les vecteurs positions
    """

    for i in range(1,n+1):
        for j in range(1,n):
            for k in range(1,p+1):  #On commence a la phase 1
                for l in range(1,n+1):
                    contraintes.append(Y[ index_Y(i,j,k,l) ] <= P[index_P(l,j,k-1)])
                    contraintes.append(Y[ index_Y(i,j,k,l) ] <= x[index(l,i,k)])
                    contraintes.append(Y[ index_Y(i,j,k,l) ] >= x[index(l,i,k)] + P[index_P(l,j,k-1)] -1)

    for i in range(1,n+1):
        for j in range(1,n):
            for k in range(1, p+1):
                contraintes.append( P[index_P(i,j,k)] == sum( Y[index_Y(i,j,k,l)]for l in range(1,n+1)))

    """Contrainte 4 : Pour chaque phase ,les bureaux initiaux d'une aile ne sont que dans une seul aile (pas plusieurs endroits)"""
    for k in range(p+1):
        for j in range(1,n) :
            contraintes.append( sum(P[index_P(i,j,k)] for i in range(1,n+1)) <= 1 )

    """Contrainte 5 : Pour chaque phase une aile ne contient que les bureaux initiaux d'une seule aile"""
    for k in range(p+1):
        for i in range(1,n+1) :
            contraintes.append( sum(P[index_P(i,j,k)] for j in range(1,n)) <= 1 )


    """ Contraintes 6 : contraintes de relaxation """
    contraintes.append(x >= 0 )
    contraintes.append(x <= 1 )
    contraintes.append(P >= 0 )
    contraintes.append(Y >= 0 )
    contraintes.append(P <= 1 )
    contraintes.append(Y <= 1 )

    """Contraintes 7 : Contraintes pour les ailes en travaux
    Si une aile est en travaux : les bureaux de cette aile doivent partir et aucune bureau ne peut venir dans cette aile"""

    #Phase 1 aile B en travaux (2)
    contraintes.append(sum( x[index(i,2,1)] * I[index(i, 2, 0)] for i in range(1, n))== 0)
    contraintes.append(sum( x[index(2,j,1)] * I[index(2, j, 0)] for j in range(1, n+1))== 1)

    #Phase 2 aile P en travaux (3)
    contraintes.append(sum( x[index(i,3,2)] * I[index(i, 3, 0)] for i in range(1, n+1))== 0)
    contraintes.append(sum( x[index(3,j,2)] * I[index(3, j, 0)] for j in range(1, n+1))== 1)

    #Phase 3 aile C/D en travaux (4)
    contraintes.append(sum( x[index(i,4,3)] * I[index(i, 4, 0)] for i in range(1, n+1))== 0)
    contraintes.append(sum( x[index(4,j,3)] * I[index(4, j, 0)] for j in range(1, n+1))== 1)


    #Phase 4 aile A en travaux
    contraintes.append(sum( x[index(i,1,4)] * I[index(i, 1, 0)] for i in range(1, n+1))== 0)
    contraintes.append(sum( x[index(1,j,4)] * I[index(1, j, 0)] for j in range(1, n+1))== 1)


    prob = cvx.Problem(function,contraintes)
    # Solving the problem
    result = prob.solve(verbose=False)

    return prob.value, P.value, x.value

val_lineaire_5, P_lineaire_5, x_lineaire_5 = modele_lineaire_5_bureaux()
print("La valeur optimale pour le probème des 5 bureaux est " + str(val_lineaire_5))

#///////////////////////////////////////////////////////////////////////////////////////////////
#MOVING PLAN



#Fonctions auxiliaires

""" Vérifie si la combinaison ne contient pas de doublons """
def est_valide(combinaison):
    return len(combinaison) == len(set(combinaison))

""" Construit et explore l'arbre des combinaisons possibles """
def construire_arbre_aux(niveaux, liste_tmp, profondeur=0, chemin_actuel=[]):
    if profondeur == len(niveaux):
        #On a trouvé une combinaison valide
        liste_tmp.append(chemin_actuel)
        return

    for valeur in niveaux[profondeur]:
        nouvelle_combinaison = chemin_actuel + [valeur]
        if est_valide(nouvelle_combinaison):
            construire_arbre_aux(niveaux, liste_tmp, profondeur + 1, nouvelle_combinaison)

def construire_arbre(liste, liste_a_retourner) :
    construire_arbre_aux(liste,liste_a_retourner )

def combinaisons_possibles(liste):
    return list(product(*liste))

"""Selon un plan de deplacement pour les 5 ailes, cette fonction calcule le nombre de deplacements"""
def calcul_deplacements(configuration) :
    compteur = 0
    for i in range (1,p+1):
        for j in range(n-1):
            if  configuration[i][j] != configuration[i-1][j] :
                compteur = compteur +1
    return compteur

"""P ne prendra pas en compte les probabilité inferieur à 0,1"""
def construction_moving_plan():
    vecteur_P= []
    borne_inferieur = val_lineaire_5
    for i in P_lineaire_5:
        if i < 10**(-1) :
            vecteur_P.append(0)
        else :
            vecteur_P.append(1)

    #Liste qui pour chaque phase donnera un plan potentiel des bureaux
    liste_des_candidats = [[] for k in range(p+1)]

    for k in range(p+1):
        liste_candidats_k = []
        for l in range(1,n):
            liste_candidats_k.append([])
        for j in range(1,n):
            listej= []
            for i in range(1,n+1):
                if vecteur_P[index_P(i,j,k)] == 1 :
                    listej.append(i)
            liste_candidats_k[j-1] = listej
        liste_des_candidats[k] = liste_candidats_k

    # Construire et explorer l'arbre pour chaque liste
    liste_candidats_possible = []
    for liste in liste_des_candidats:
        liste_combinaisons_possible = []
        construire_arbre(liste, liste_combinaisons_possible)
        liste_candidats_possible.append(liste_combinaisons_possible)

    combinaisons = combinaisons_possibles(liste_candidats_possible)

    nb = 1
    trouve = False
    plan = None
    for combinaison in combinaisons :
        if (calcul_deplacements(combinaison) <=  borne_inferieur + 1):
            if trouve == False :
                trouve = True
                plan = combinaison
            #On a trouvé un plan de deplacement de cout minimal
            print("Plan de deplacement possible n° " + str(nb) )
            nb = nb+1
            for phase in combinaison :
                print(phase)
    return plan

def detail_des_deplacements1(plan):
    correspondance = {1: "A1", 2: "B1", 3: "P1", 4: "C1", 5: "N1"}

    for i in range(1, len(plan)):
        phase_actuelle = plan[i]
        phase_precedente = plan[i-1]
        print("Phase " + str(i))

        for j in range(len(phase_actuelle)):
            bureau_precedent = correspondance[phase_precedente[j]]
            bureau_actuel = correspondance[phase_actuelle[j]]

            if phase_actuelle[j] != phase_precedente[j]:
                print("Les bureaux en " + bureau_precedent + " se déplacent vers les bureaux en " + bureau_actuel)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#Deuxieme problème vec les 13 bureaux

#x = (x110, x111, ..., x11p, x120, ..x12p, ..., x1n0, ...x1np, x210, ....xnnp)
p = 5  #nombre de phases
n = 13 #nombre de bureaux

def index_Y(i,j,k,l):
    return (i-1) * (n-3) * n * (p+1) + (j-1) * n * (p+1) + k * n + l-1

def index_P(i,j,k):
    return   (i - 1) * (n-3) * (p + 1) + (j - 1) * (p + 1) + k


def modele_lineaire_13_bureaux():
    x = cvx.Variable((n*n*(p+1)))
    P = cvx.Variable((n*(n-3)*(p+1)))
    Y = cvx.Variable((n*(n-3)*n*(p+1)))
    I = give_equality()

    function = cvx.Minimize(x.T @ I)
    contraintes = []

    #x entre 0 et 1 (relaxation)
    contraintes.append(x >= 0)
    contraintes.append(x <= 1)
    contraintes.append(P >= 0)
    contraintes.append(Y >= 0)

    """Contrainte 1 : Initialisation """

    for i in range(1,n-2):
        contraintes.append(P[index_P(i,i,0)] == 1)
    contraintes.append(sum( sum( x[index(i,j,0)] * I[index(i, j, 0)]  for j in range(1,n+1) ) for i in range(1,n+1)) == 0)

    """ Contrainte 2 : Les emplacements finaux sont imposés """

    contraintes.append( P[index_P(1,1,p)] == 1 )
    contraintes.append( P[index_P(5,2,p)] == 1 )
    contraintes.append( P[index_P(7,6,p)] == 1 )

    contraintes.append( P[index_P(13,3,p)] == 1 )
    contraintes.append( P[index_P(6,4,p)] == 1 )
    contraintes.append( P[index_P(10,10,p)] == 1 )

    contraintes.append( P[index_P(8,5,p)] == 1 )
    contraintes.append( P[index_P(9,7,p)] == 1 )
    contraintes.append( P[index_P(11,8,p)] == 1 )
    contraintes.append( P[index_P(12,9,p)] == 1 )

    """Contrainte 3 : Ailes en travaux """

    #Phase 1 Aile B en travaux (bureaux 4,6,10)
    liste_indices = [4,6,10]
    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,1)] * I[index(i, j, 0)] for i in range(1, n+1)) == 0)
        contraintes.append(sum( x[index(j,i,1)] * I[index(j, i, 0)] for i in range(1, n+1)) == 1)

    #Phase 2 Aile P en travaux (bureaux 1,5,7)
    liste_indices = [1,5,7]

    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,2)] * I[index(i, j, 0)] for i in range(1, n+1))== 0)
        contraintes.append(sum( x[index(j,i,2)] * I[index(j, i, 0)] for i in range(1, n+1))== 1)


    #Phase 3 Aile C en travaux (bureaux 8,9)
    liste_indices = [8,9]

    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,3)] * I[index(i, j, 0)] for i in range(1, n+1)) == 0)
        contraintes.append(sum( x[index(j,i,3)] * I[index(j, i, 0)] for i in range(1, n+1)) == 1)


    #Phase 4 Aile A en travaux (bureaux 2,3)
    liste_indices = [2,3]

    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,4)] * I[index(i, j, 0)] for i in range(1, n+1))== 0)
        contraintes.append(sum( x[index(j,i,4)] * I[index(j, i, 0)] for i in range(1, n+1))== 1)


    """Contrainte 4 : un mouvement max par bureau """

    for k in range(1,p+1):
        for i in range(1,n+1):
            contraintes.append(sum(x[index(i,j,k)] for j in range(1,n+1)) == 1)


    """Contraintes 5 : Contraintes linéaires pour P"""
    #On a ajouté des contraintes linéaires pour affecter les valeurs de positions
    for i in range(1,n+1):
        for j in range(1,n-2):
            for k in range(1,p+1):  #On commence a la phase 1
                for l in range(1,n+1):
                    contraintes.append(Y[ index_Y(i,j,k,l) ] <= P[index_P(l,j,k-1)])
                    contraintes.append(Y[ index_Y(i,j,k,l) ] <= x[index(l,i,k)])
                    contraintes.append(Y[ index_Y(i,j,k,l) ] >= x[index(l,i,k)] + P[index_P(l,j,k-1)] -1)

    for i in range(1,n+1):
        for j in range(1,n-2):
            for k in range(p+1):
                contraintes.append( P[index_P(i,j,k)] == sum( Y[index_Y(i,j,k,l)]for l in range(1,n+1)))

    """Contrainte 6 : Pour chaque phase ,les bureaux initiaux d'une aile ne sont que dans une seul aile (pas plusieurs endroits)"""
    for k in range(p+1):
        for j in range(1,n-2) :
            contraintes.append( sum(P[index_P(i,j,k)] for i in range(1,n+1)) <= 1 )

    """Contrainte 7 : Pour chaque phase une aile ne contient que les bureaux initiaux d'une seule aile"""
    for k in range(p+1):
        for i in range(1,n+1) :
            contraintes.append( sum(P[index_P(i,j,k)] for j in range(1,n-2)) <= 1 )


    prob = cvx.Problem(function,contraintes)

    # Solving the problem

    result = prob.solve(verbose=False)
    return prob.value, P.value, x.value


val_lineaire_13,  P_lineaire_13, x_lineaire_13 = modele_lineaire_13_bureaux()
print("La valeur optimale pour le probème des 5 bureaux est " + str(val_lineaire_13))


#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#MOVING PLAN POUR LES 13 BUREAUX 

#Selon un plan de deplacement pour les 5 ailes, cette fonction calcule le nombre de deplacements
def calcul_deplacements(configuration) :
    """ est differente de calcul_deplacements(configuration) pour le problème à 5 bureaux """
    compteur = 0
    for i in range (1,p+1):
        for j in range(n-3):
            if  configuration[i][j] != configuration[i-1][j] :
                compteur = compteur +1
    return compteur
def construction_moving_plan_13():
    vecteur_P = []
    borne_inferieur = val_lineaire_13
    for i in P_lineaire_13 :
        if i < 10**(-1) :
            vecteur_P.append(0)
        else :
            vecteur_P.append(1)

    liste_des_candidats = [[] for k in range(p+1)]

    for k in range(p+1):
        liste_candidats_k = []
        for l in range(1,n-2):
            liste_candidats_k.append([])

        for j in range(1,n-2):
            listej= []
            for i in range(1,n+1):
                if vecteur_P[index_P(i,j,k)] == 1 :
                    listej.append(i)
            liste_candidats_k[j-1] = listej
        liste_des_candidats[k] = liste_candidats_k

    # Construire et explorer l'arbre pour chaque liste
    liste_candidats_possible = []
    for liste in liste_des_candidats:
        liste_combinaisons_possible = []
        construire_arbre(liste, liste_combinaisons_possible)
        liste_candidats_possible.append(liste_combinaisons_possible)

    combinaisons = combinaisons_possibles(liste_candidats_possible)

    nb = 1
    trouve = False
    plan = None
    for combinaison in combinaisons :
        if (calcul_deplacements(combinaison) <=  borne_inferieur + 1):
            #On a trouvé un plan de deplacement de cout minimal
            if trouve == False :
                trouve = True
                plan = combinaison
            print("Plan de deplacement possible n° " + str(nb) )
            nb = nb+1
            for phase in combinaison :
                print(phase)
    return plan


def detail_des_deplacements_13Bis(plan):
    correspondance_numeros = {
        "P3": 1,
        "A1": 2,
        "A2": 3,
        "B3": 4,
        "P2": 5,
        "B2": 6,
        "P1": 7,
        "C1": 8,
        "C2": 9,
        "B1": 10,
        "N1": 11,
        "N2": 12,
        "N3": 13
    }

    correspondance_etiquettes = {v: k for k, v in correspondance_numeros.items()}

    for i in range(1, len(plan)):
        phase_actuelle = plan[i]
        phase_precedente = plan[i - 1]
        print("Phase " + str(i))

        for j in range(len(phase_actuelle)):
            bureau_precedent = correspondance_etiquettes[phase_precedente[j]]
            bureau_actuel = correspondance_etiquettes[phase_actuelle[j]]

            if phase_actuelle[j] != phase_precedente[j]:
                print("Les bureaux en " + bureau_precedent + " se déplacent vers les bureaux en " + bureau_actuel)



plan = construction_moving_plan_13()




#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#CONSTRUCTION DU GRAPH

n = 13
p = 5
bureau_mapping = {
    1: 'P3', 2: 'A1', 3: 'A2', 4: 'B3',
    5: 'P2', 11: 'N1', 12: 'N2', 13: 'N3', 6: 'B2',
    7: 'P1', 8: 'C1', 9: 'C2', 10: 'B1'
}

nodes = ['P3', 'A1','A2', 'B3' , 'P2','N1', 'N2', 'N3', 'B2',  'P1', 'C1', 'C2', 'B1']
edges = [('P3', 'A1'), ('A1', 'A2'), ('A2', 'B3'), ('B3', 'B2'), ('B2', 'B1'), ('B1', 'C2'), ('C2', 'C1'),('C1', 'P1'), ('P1', 'P2'), ('P2', 'P3'), ('P2', 'N1'), ('N2', 'N3'),
                  ('N2', 'N3'), ('N3', 'B2')]

pos = {
        'P3': (0, 2),
        'A1': (1.5, 2),
        'A2': (2.6, 2),
        'B3': (4, 2),

        'P2': (0, 1),
        'N1': (1, 1),
        'N2': (2, 1),
        'N3': (3, 1),
        'B2': (4, 1),

        'P1': (0, 0),
        'C1': (1.5, 0),
        'C2': (2.6, 0),
        'B1': (4, 0),
    }

def update_edges(position) :
    liste_ligne = [dict(), dict(), dict()]
    for i in position.keys() :
        c,l = position[i]
        liste_ligne[l][i] = c,l
    for i in range(len(liste_ligne)) :
        liste_ligne[i]= dict(sorted(liste_ligne[i].items(), key=lambda item: item[1][0]))

    i = 0
    for dico in liste_ligne :
        liste_ligne[i] = list(dico.keys())
        i+=1
    #Ligne 0
    new_edges = []
    for i in range(3):
        new_edges.append((liste_ligne[0][i] , liste_ligne[0][i+1] ))

    new_edges.append((liste_ligne[0][0], liste_ligne[1][0] ))
    new_edges.append((liste_ligne[0][3], liste_ligne[1][4]))

    #Ligne 1
    for i in range(4):
        new_edges.append((liste_ligne[1][i], liste_ligne[1][i+1]))
    new_edges.append((liste_ligne[1][0], liste_ligne[2][0]))
    new_edges.append((liste_ligne[1][4], liste_ligne[2][3]))

    #ligne 2
    for i in range(3):
        new_edges.append((liste_ligne[2][i], liste_ligne[2][i+1]))

    return new_edges


def create_graph(pos, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue')
    plt.show()

def new_configuration(configuration, position, k):
    edges = []
    new_pos = dict()
    ancienne_config = configuration[k-1]
    nouvelle_config = configuration[k]
    vide = []

    for i in range(10):
        new_pos[bureau_mapping[i+1]] = pos[bureau_mapping[nouvelle_config[i]]]
    tout = [ i for i in range(1,14)]

    vide = list(filter(lambda x: x not in nouvelle_config, tout))
    empty = ['N1', 'N2', 'N3']

    for i in range(len(vide)) :
        new_pos[empty[i]] =  pos[bureau_mapping[vide[i]]]
    return new_pos


def graph_movement(configuration) :
    position = pos
    for k in range(p+1) :
        print("PHASE " + str(k))
        if k == 0 :
            create_graph(pos,edges)
        else :
            position = new_configuration(configuration, position, k)
            new_edges = update_edges(position)
            create_graph(position, new_edges)


configuration = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [1, 2, 3, 11, 5, 9, 7, 8, 12, 13],
                [8, 2, 13, 6, 3, 9, 4, 11, 12, 10],
                [1, 2, 13, 6, 3, 7, 4, 11, 12, 10],
                [1, 5, 13, 6, 8, 7, 9, 11, 12, 10],
                [1, 5, 13, 6, 8, 7, 9, 11, 12, 10]]

graph_movement(configuration)



#////////////////////////////////////////////////////////////////////////
#EVITER DE REGROUPER STUDENTS ET PRESIDENCE 
p = 5  #nombre de phases
n = 13 #nombre de bureaux

def index_Y(i,j,k,l):
    return (i-1) * (n-3) * n * (p+1) + (j-1) * n * (p+1) + k * n + l-1

def index_P(i,j,k):
    return   (i - 1) * (n-3) * (p + 1) + (j - 1) * (p + 1) + k

def modele_lineaire_eviter_regroupement():
    x = cvx.Variable((n*n*(p+1)))
    P = cvx.Variable((n*(n-3)*(p+1)))
    Y = cvx.Variable((n*(n-3)*n*(p+1)))
    W = cvx.Variable(((n-2) * p * 2*4 + 2*p*3*4))
    I = give_equality()

    function = cvx.Minimize(x.T @ I)
    contraintes = []

    #x entre 0 et 1 (relaxation)
    contraintes.append(x >= 0)
    contraintes.append(x <= 1)
    contraintes.append(P >= 0)
    contraintes.append(Y >= 0)

    """Contrainte 1 : Initialisation """

    for i in range(1,n-2):
        contraintes.append(P[index_P(i,i,0)] == 1)
    contraintes.append(sum( sum( x[index(i,j,0)] * I[index(i, j, 0)]  for j in range(1,n+1) ) for i in range(1,n+1)) == 0)

    """ Contrainte 2 : Les emplacements finaux sont imposés """

    contraintes.append( P[index_P(1,1,p)] == 1 )
    contraintes.append( P[index_P(5,2,p)] == 1 )
    contraintes.append( P[index_P(7,6,p)] == 1 )

    contraintes.append( P[index_P(13,3,p)] == 1 )
    contraintes.append( P[index_P(6,4,p)] == 1 )
    contraintes.append( P[index_P(10,10,p)] == 1 )

    contraintes.append( P[index_P(8,5,p)] == 1 )
    contraintes.append( P[index_P(9,7,p)] == 1 )
    contraintes.append( P[index_P(11,8,p)] == 1 )
    contraintes.append( P[index_P(12,9,p)] == 1 )

    """Contrainte 3 : Ailes en travaux """

    #Phase 1 Aile B en travaux (bureaux 4,6,10)
    liste_indices = [4,6,10]
    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,1)] * I[index(i, j, 0)] for i in range(1, n+1)) == 0)
        contraintes.append(sum( x[index(j,i,1)] * I[index(j, i, 0)] for i in range(1, n+1)) == 1)

    #Phase 2 Aile P en travaux (bureaux 1,5,7)
    liste_indices = [1,5,7]

    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,2)] * I[index(i, j, 0)] for i in range(1, n+1))== 0)
        contraintes.append(sum( x[index(j,i,2)] * I[index(j, i, 0)] for i in range(1, n+1))== 1)


    #Phase 3 Aile C en travaux (bureaux 8,9)
    liste_indices = [8,9]

    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,3)] * I[index(i, j, 0)] for i in range(1, n+1)) == 0)
        contraintes.append(sum( x[index(j,i,3)] * I[index(j, i, 0)] for i in range(1, n+1)) == 1)


    #Phase 4 Aile A en travaux (bureaux 2,3)
    liste_indices = [2,3]

    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,4)] * I[index(i, j, 0)] for i in range(1, n+1))== 0)
        contraintes.append(sum( x[index(j,i,4)] * I[index(j, i, 0)] for i in range(1, n+1))== 1)


    """Contrainte 4 : un mouvement max par bureau """

    for k in range(1,p+1):
        for i in range(1,n+1):
            contraintes.append(sum(x[index(i,j,k)] for j in range(1,n+1)) == 1)


    """Contraintes 5 : Contraintes linéaires pour P"""
    #On a ajouté des contraintes linéaires pour affecter les valeurs de positions
    for i in range(1,n+1):
        for j in range(1,n-2):
            for k in range(1,p+1):  #On commence a la phase 1
                for l in range(1,n+1):
                    contraintes.append(Y[ index_Y(i,j,k,l) ] <= P[index_P(l,j,k-1)])
                    contraintes.append(Y[ index_Y(i,j,k,l) ] <= x[index(l,i,k)])
                    contraintes.append(Y[ index_Y(i,j,k,l) ] >= x[index(l,i,k)] + P[index_P(l,j,k-1)] -1)

    for i in range(1,n+1):
        for j in range(1,n-2):
            for k in range(p+1):
                contraintes.append( P[index_P(i,j,k)] == sum( Y[index_Y(i,j,k,l)]for l in range(1,n+1)))

    """Contrainte 6 : Pour chaque phase ,les bureaux initiaux d'une aile ne sont que dans une seul aile (pas plusieurs endroits)"""
    for k in range(p+1):
        for j in range(1,n-2) :
            contraintes.append( sum(P[index_P(i,j,k)] for i in range(1,n+1)) <= 1 )

    """Contrainte 7 : Pour chaque phase une aile ne contient que les bureaux initiaux d'une seule aile"""
    for k in range(p+1):
        for i in range(1,n+1) :
            contraintes.append( sum(P[index_P(i,j,k)] for j in range(1,n-2)) <= 1 )


    #clé : un numero d'office
    #valeur : les offices qui se trouvent à coté de lui

    near_to = dict()
    near_to[1] = [2,5]
    near_to[2] = [1,3]
    near_to[3] = [2,4]
    near_to[4] = [3,6]
    near_to[5] = [1,7,11]
    near_to[6] = [4,10,13]
    near_to[7] = [5,8]
    near_to[8] = [7,9]
    near_to[9] = [8,10]
    near_to[10] = [6,9]
    near_to[11] = [5,12]
    near_to[12] = [11,13]
    near_to[13] = [12,6]

    ind1 = 1
    phase = 1
    m = 0

    for i in range(1,n+1):
        for k in range(1,p+1):
            for cote in near_to[i]:
                contraintes.append(W[m] <= P[index_P(i,8,k)] )
                contraintes.append(W[m] <= P[index_P(cote,5,k)])
                contraintes.append(W[m] >= P[index_P(i,8,k)] + P[index_P(cote,5,k)] -1)
                m+=1
                contraintes.append(W[m] <= P[index_P(i,8,k)])
                contraintes.append(W[m] <=  P[index_P(cote,7,k)])
                contraintes.append(W[m] >= P[index_P(i,8,k)] + P[index_P(cote,7,k)] -1 )
                m+=1
                contraintes.append(W[m] <= P[index_P(i,9,k)] )
                contraintes.append(W[m] <= P[index_P(cote,5,k)])
                contraintes.append(W[m] >= P[index_P(i,9,k)] +P[index_P(cote,5,k)] -1 )
                m+=1
                contraintes.append(W[m] <= P[index_P(i,9,k)] )
                contraintes.append(W[m] <= P[index_P(cote,7,k)] )
                contraintes.append(W[m] >= P[index_P(i,9,k)]+ P[index_P(cote,7,k)] -1 )
                m+=1

    for i in range(((n-2) * p * 2*4 + 2*p*3*4)) :
        contraintes.append(W[i] == 0 )

    prob = cvx.Problem(function,contraintes)
    # Solving the problem
    result = prob.solve(verbose=False)

    return prob.value, P.value, x.value

#A EXECUTER 
#val_lineaire_13_regroupement,  P_lineaire_13_regroupement, x_lineaire_13_regroupement = modele_lineaire_eviter_regroupement()
#print("La valeure optimale pour les 13 bureaux en evitant les regroupements de 'presidency' et 'students association est : "+str(val_lineaire_13_regroupement))


#////////////////////////////////////////////////////////////////////////
#Partie 2 Question 1

#x = (x110, x111, ..., x11p, x120, ..x12p, ..., x1n0, ...x1np, x210, ....xnnp)
p = 5 #nombre de phases
n = 13 #nombre de bureaux

def index_Y(i,j,k,l):
    return (i-1) * (n-3) * n * (p+1) + (j-1) * n * (p+1) + k * n + l-1

def index_P(i,j,k):
    return   (i - 1) * (n-3) * (p + 1) + (j - 1) * (p + 1) + k

def modele_quadratique(l):

    x = cvx.Variable((n*n*(p+1)))
    P = cvx.Variable((n*(n-3)*(p+1)))
    P_penalty = cvx.Variable((n*(n-3)*(p+1)))
    Y = cvx.Variable((n*(n-3)*n*(p+1)))
    I = give_equality()



    function = cvx.Minimize(x.T @ I+ l*(cvx.norm(P_penalty, 2))**2)
    contraintes = []

    #x entre 0 et 1 (relaxation)
    contraintes.append(x >= 0)
    contraintes.append(x <= 1)
    contraintes.append(P >= 0)
    contraintes.append(Y >= 0)

    """Contrainte 1 : Initialisation """

    for i in range(1,n-2):
        contraintes.append(P[index_P(i,i,0)] == 1)
    contraintes.append(sum( sum( x[index(i,j,0)] * I[index(i, j, 0)]  for j in range(1,n+1) ) for i in range(1,n+1)) == 0)

    """ Contrainte 2 : Les emplacements finaux sont imposés """

    contraintes.append( P[index_P(1,1,p)] == 1 )
    contraintes.append( P[index_P(5,2,p)] == 1 )
    contraintes.append( P[index_P(7,6,p)] == 1 )

    contraintes.append( P[index_P(13,3,p)] == 1 )
    contraintes.append( P[index_P(6,4,p)] == 1 )
    contraintes.append( P[index_P(10,10,p)] == 1 )

    contraintes.append( P[index_P(8,5,p)] == 1 )
    contraintes.append( P[index_P(9,7,p)] == 1 )
    contraintes.append( P[index_P(11,8,p)] == 1 )
    contraintes.append( P[index_P(12,9,p)] == 1 )

    """Contrainte 3 : Ailes en travaux """

    #Phase 1 Aile B en travaux (bureaux 4,6,10)
    liste_indices = [4,6,10]
    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,1)] * I[index(i, j, 0)] for i in range(1, n+1)) == 0)
        contraintes.append(sum( x[index(j,i,1)] * I[index(j, i, 0)] for i in range(1, n+1)) == 1)

    #Phase 2 Aile P en travaux (bureaux 1,5,7)
    liste_indices = [1,5,7]

    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,2)] * I[index(i, j, 0)] for i in range(1, n+1))== 0)
        contraintes.append(sum( x[index(j,i,2)] * I[index(j, i, 0)] for i in range(1, n+1))== 1)


    #Phase 3 Aile C en travaux (bureaux 8,9)
    liste_indices = [8,9]

    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,3)] * I[index(i, j, 0)] for i in range(1, n+1)) == 0)
        contraintes.append(sum( x[index(j,i,3)] * I[index(j, i, 0)] for i in range(1, n+1)) == 1)


    #Phase 4 Aile A en travaux (bureaux 2,3)
    liste_indices = [2,3]

    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,4)] * I[index(i, j, 0)] for i in range(1, n+1))== 0)
        contraintes.append(sum( x[index(j,i,4)] * I[index(j, i, 0)] for i in range(1, n+1))== 1)


    """Contrainte 4 : un mouvement max par bureau """

    for k in range(1,p+1):
        for i in range(1,n+1):
            contraintes.append(sum(x[index(i,j,k)] for j in range(1,n+1)) == 1)

    """Contraintes 5 : Contraintes linéaires pour P"""

    for i in range(1,n+1):
        for j in range(1,n-2):
            for k in range(1,p+1):  #On commence a la phase 1
                for l in range(1,n+1):
                    contraintes.append(Y[ index_Y(i,j,k,l) ] <= P[index_P(l,j,k-1)])
                    contraintes.append(Y[ index_Y(i,j,k,l) ] <= x[index(l,i,k)])
                    contraintes.append(Y[ index_Y(i,j,k,l) ] >= x[index(l,i,k)] + P[index_P(l,j,k-1)] -1)

    for i in range(1,n+1):
        for j in range(1,n-2):
            for k in range(p+1):
                contraintes.append( P[index_P(i,j,k)] == sum( Y[index_Y(i,j,k,l)]for l in range(1,n+1)))

    """Contrainte 6 : Pour chaque phase ,les bureaux initiaux d'une aile ne sont que dans une seul aile (pas plusieurs endroits)"""
    for k in range(p+1):
        for j in range(1,n-2) :
            contraintes.append( sum(P[index_P(i,j,k)] for i in range(1,n+1)) <= 1 )

    """Contrainte 7 : Pour chaque phase une aile ne contient que les bureaux initiaux d'une seule aile"""
    for k in range(p+1):
        for i in range(1,n+1) :
            contraintes.append( sum(P[index_P(i,j,k)] for j in range(1,n-2)) <= 1)

    for k in range(p):
        for i in range(1,n+1):
            for j in range(1,n-2):
                contraintes.append(P_penalty[index_P(i,j,k)] == P[index_P(i,j,k)] - P[index_P(i,j,p)])

    prob = cvx.Problem(function  ,contraintes)

    # Solving the problem
    result = prob.solve(verbose=False)

    return prob.value, P.value, x.value



#////////////////////////////////////////////////////////////////////////
# A executer 
#I = give_equality()


#val_quadratique_1, P_quadratique_1, x_quadratique_1 = modele_quadratique(1)
#val_quadratique_100, P_quadratique_100, x_quadratique_100 = modele_quadratique(100)

#print("La valeure optimale pour le problème quadratique pour lambda = 1 est : "+str(val_quadratique_1))
#print("nombre de deplacements pour lambda = 1  :" + str(x_quadratique_1.T @ I))
#print("La valeure optimale pour le problème quadratique pour lambda = 100 est : "+str(val_quadratique_100))
#print("nombre de deplacements pour lambda = 100 : " +str(x_quadratique_100.T @ I))




#////////////////////////////////////////////////////////////////////////
#Calcul du nombre minimal de deplacements pour l = 1 et 100


#Selon un plan de deplacement pour les 5 ailes, cette fonction calcule le nombre de deplacements
def calcul_deplacements(configuration) :
    compteur = 0
    for i in range (1,p+1):
        for j in range(n-3):
            if  configuration[i][j] != configuration[i-1][j] :
                compteur = compteur +1
    return compteur

def calcul_nombre_minimal_deplacement(vecteurP) :
    vecteur_P = []
    for i in vecteurP :
        if i < 10**(-1) :
            vecteur_P.append(0)
        else :
            vecteur_P.append(1)

    liste_des_candidats = [[] for k in range(p+1)]

    for k in range(p+1):
        liste_candidats_k = []
        for l in range(1,n-2):
            liste_candidats_k.append([])

        for j in range(1,n-2):
            listej= []
            for i in range(1,n+1):
                if vecteur_P[index_P(i,j,k)] == 1 :
                    listej.append(i)
            liste_candidats_k[j-1] = listej
        liste_des_candidats[k] = liste_candidats_k

    # Construire et explorer l'arbre pour chaque liste
    liste_candidats_possible = []
    for liste in liste_des_candidats:
        liste_combinaisons_possible = []
        construire_arbre(liste, liste_combinaisons_possible)
        liste_candidats_possible.append(liste_combinaisons_possible)

    combinaisons = combinaisons_possibles(liste_candidats_possible)

    nombre_minimal_deplacements = 1000000
    for combinaison in combinaisons :
        nbdeplacements = calcul_deplacements(combinaison)
        if (nbdeplacements <  nombre_minimal_deplacements):
            nombre_minimal_deplacements = nbdeplacements
    return nombre_minimal_deplacements


#A executer 
#print("nombre minimal de deplacements pour lambda = 1 : " + str(calcul_nombre_minimal_deplacement(P_quadratique_1)) )
#print("nombre minimal de deplacements pour lambda = 100 : " + str(calcul_nombre_minimal_deplacement(P_quadratique_100)) )



#////////////////////////////////////////////////////////////////////////
#Penalité avec le vecteur initial 
#x = (x110, x111, ..., x11p, x120, ..x12p, ..., x1n0, ...x1np, x210, ....xnnp)
p = 5 #nombre de phases
n = 13 #nombre de bureaux

def index_Y(i,j,k,l):
    return (i-1) * (n-3) * n * (p+1) + (j-1) * n * (p+1) + k * n + l-1

def index_P(i,j,k):
    return   (i - 1) * (n-3) * (p + 1) + (j - 1) * (p + 1) + k

def modele_quadratique_vecteur_initial(l):

    x = cvx.Variable((n*n*(p+1)))
    P = cvx.Variable((n*(n-3)*(p+1)))
    P_penalty = cvx.Variable((n*(n-3)*(p+1)))
    Y = cvx.Variable((n*(n-3)*n*(p+1)))
    I = give_equality()



    function = cvx.Minimize(x.T @ I+ l*(cvx.norm(P_penalty, 2))**2)
    contraintes = []

    #x entre 0 et 1 (relaxation)
    contraintes.append(x >= 0)
    contraintes.append(x <= 1)
    contraintes.append(P >= 0)
    contraintes.append(Y >= 0)

    """Contrainte 1 : Initialisation """

    for i in range(1,n-2):
        contraintes.append(P[index_P(i,i,0)] == 1)
    contraintes.append(sum( sum( x[index(i,j,0)] * I[index(i, j, 0)]  for j in range(1,n+1) ) for i in range(1,n+1)) == 0)

    """ Contrainte 2 : Les emplacements finaux sont imposés """

    contraintes.append( P[index_P(1,1,p)] == 1 )
    contraintes.append( P[index_P(5,2,p)] == 1 )
    contraintes.append( P[index_P(7,6,p)] == 1 )

    contraintes.append( P[index_P(13,3,p)] == 1 )
    contraintes.append( P[index_P(6,4,p)] == 1 )
    contraintes.append( P[index_P(10,10,p)] == 1 )

    contraintes.append( P[index_P(8,5,p)] == 1 )
    contraintes.append( P[index_P(9,7,p)] == 1 )
    contraintes.append( P[index_P(11,8,p)] == 1 )
    contraintes.append( P[index_P(12,9,p)] == 1 )

    """Contrainte 3 : Ailes en travaux """

    #Phase 1 Aile B en travaux (bureaux 4,6,10)
    liste_indices = [4,6,10]
    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,1)] * I[index(i, j, 0)] for i in range(1, n+1)) == 0)
        contraintes.append(sum( x[index(j,i,1)] * I[index(j, i, 0)] for i in range(1, n+1)) == 1)

    #Phase 2 Aile P en travaux (bureaux 1,5,7)
    liste_indices = [1,5,7]

    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,2)] * I[index(i, j, 0)] for i in range(1, n+1))== 0)
        contraintes.append(sum( x[index(j,i,2)] * I[index(j, i, 0)] for i in range(1, n+1))== 1)


    #Phase 3 Aile C en travaux (bureaux 8,9)
    liste_indices = [8,9]

    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,3)] * I[index(i, j, 0)] for i in range(1, n+1)) == 0)
        contraintes.append(sum( x[index(j,i,3)] * I[index(j, i, 0)] for i in range(1, n+1)) == 1)


    #Phase 4 Aile A en travaux (bureaux 2,3)
    liste_indices = [2,3]

    for j in liste_indices :
        contraintes.append(sum( x[index(i,j,4)] * I[index(i, j, 0)] for i in range(1, n+1))== 0)
        contraintes.append(sum( x[index(j,i,4)] * I[index(j, i, 0)] for i in range(1, n+1))== 1)


    """Contrainte 4 : un mouvement max par bureau """

    for k in range(1,p+1):
        for i in range(1,n+1):
            contraintes.append(sum(x[index(i,j,k)] for j in range(1,n+1)) == 1)

    """Contraintes 5 : Contraintes linéaires pour P"""

    for i in range(1,n+1):
        for j in range(1,n-2):
            for k in range(1,p+1):  #On commence a la phase 1
                for l in range(1,n+1):
                    contraintes.append(Y[ index_Y(i,j,k,l) ] <= P[index_P(l,j,k-1)])
                    contraintes.append(Y[ index_Y(i,j,k,l) ] <= x[index(l,i,k)])
                    contraintes.append(Y[ index_Y(i,j,k,l) ] >= x[index(l,i,k)] + P[index_P(l,j,k-1)] -1)

    for i in range(1,n+1):
        for j in range(1,n-2):
            for k in range(p+1):
                contraintes.append( P[index_P(i,j,k)] == sum( Y[index_Y(i,j,k,l)]for l in range(1,n+1)))

    """Contrainte 6 : Pour chaque phase ,les bureaux initiaux d'une aile ne sont que dans une seul aile (pas plusieurs endroits)"""
    for k in range(p+1):
        for j in range(1,n-2) :
            contraintes.append( sum(P[index_P(i,j,k)] for i in range(1,n+1)) <= 1 )

    """Contrainte 7 : Pour chaque phase une aile ne contient que les bureaux initiaux d'une seule aile"""
    for k in range(p+1):
        for i in range(1,n+1) :
            contraintes.append( sum(P[index_P(i,j,k)] for j in range(1,n-2)) <= 1)

    for k in range(p):
        for i in range(1,n+1):
            for j in range(1,n-2):
                contraintes.append(P_penalty[index_P(i,j,k)] == P[index_P(i,j,k)] - P[index_P(i,j,0)])

    prob = cvx.Problem(function  ,contraintes)

    # Solving the problem
    result = prob.solve(verbose=False)
    return prob.value, P.value, x.value


#A executer 
#I = give_equality()
#val_quadratique_initial_1, P_quadratique_initial_1, x_quadratique_initial_1 = modele_quadratique_vecteur_initial(1)
#val_quadratique_initial_100, P_quadratique_initial_100, x_quadratique_initial_100 = modele_quadratique_vecteur_initial(100)


#print("La valeure optimale pour le problème quadratique avec le vecteur initial  pour lambda = 1 est : "+str(val_quadratique_initial_1))
#print("nombre de deplacements pour lambda = 1 : " + str(x_quadratique_initial_1.T @ I))
#print("La valeure optimale pour le problème quadratique avec le vecteur initial  pour lambda = 100 est : "+str(val_quadratique_initial_100))

#print("nombre de deplacements pour lambda = 100: " + str(x_quadratique_initial_100.T @ I))





#////////////////////////////////////////////////////////////////////////
#SDP
p = 5  # nombre de phases
n = 5  # nombre d'ailes

x_dimension = (n*n*(p+1))
P_dimension = (n*(n-1)*(p+1))
u_dimension = x_dimension + P_dimension

#récupère l'indice du vecteur qui correspond au déplacement des bureaux de l'aile i à l'aile j lors de la phase p
def index(i,j,k):
    return (i - 1) * n * (p + 1) + (j - 1) * (p + 1) + k

def index_Y(i,j,k,l):
    return (i-1) * (n-1) * n * (p+1) + (j-1) * n * (p+1) + k * n + l-1

def index_P(i,j,k):
    return   (i - 1) * (n-1) * (p + 1) + (j - 1) * (p + 1) + k

def index_u(choix, i,j,k):
    if choix == 1 :
        return index(i,j,k)
    else :
        return x_dimension + index_P(i,j,k)


#créé le vecteur tel qu'il y a 1 lorsque les bureaux se déplacent et 0 pour toutes les fois où les bureaux restent sur place
def give_equality():
    I = np.ones(n*n*(p+1))
    for i in range(n):
        for k in range(p+1):
            ind = index(i+1,i+1,k)
            I[ind] = 0
    return I

def give_equality_SDP():
  I = np.eye(x_dimension)
  for i in range(1,n+1):
      for k in range(p+1):
        I[index(i,i,k), index(i,i,k)] = 0
  return I

def modele_SDP_5():

    x = cvx.Variable(x_dimension)
    I = give_equality()
    I2 = give_equality_SDP()
    P = cvx.Variable(P_dimension)
    u = cvx.Variable((x_dimension + P_dimension))
    U = cvx.Variable((x_dimension + P_dimension, x_dimension + P_dimension))
    Y = cvx.Variable((x_dimension + P_dimension +1, x_dimension + P_dimension+1))
    z = cvx.Variable(1)

    function = cvx.Minimize(cvx.trace( cvx.diag(x) @ I2) )
    contraintes = []

   #///////////CONTRAINTES SUR X ET P ////////

    """Initialisation des positions """
    for i in range(1,n):
        contraintes.append(P[index_P(i,i,0)] == 1)
        contraintes.append(P[index_P(i,i,5)] == 1)


    """A la phase 0 personne ne bouge """
    contraintes.append(sum( sum( x[index(i,j,0)] * I[index(i, j, 0)]  for j in range(1,n+1) ) for i in range(1,n+1)) == 0)

    """Pour chaque phase il y a un mouvement dans chaque aile"""
    for k in range(1,p+1):
        if (k == 1):
            for i in range(1,n) :
                contraintes.append( sum( x[index(i,j,k)] for j in range(1,n+1)) == 1)
            contraintes.append(sum( x[index(n,j,k)] for j in range(1,n+1)) == 0)
        else :
            for i in range(1,n+1):
                contraintes.append( sum( x[index(i,j,k)] for j in range(1,n+1)) == 1)

    """ Pour chaque phase ,les bureaux initiaux d'une aile ne sont que dans une seul aile (pas plusieurs endroits)"""
    for k in range(p+1):
        for j in range(1,n) :
            contraintes.append( sum(P[index_P(i,j,k)] for i in range(1,n+1)) <= 1 )

    """ Pour chaque phase une aile ne contient que les bureaux initiaux d'une seule aile"""
    for k in range(p+1):
        for i in range(1,n+1) :
            contraintes.append( sum(P[index_P(i,j,k)] for j in range(1,n)) <= 1 )

    """Contraintes de relaxation """
    contraintes.append(x >= 0 )
    contraintes.append(x <= 1 )
    contraintes.append(P >= 0 )
    contraintes.append(P <= 1 )

    """Contraintes pour les ailes en travaux
    Si une aile est en travaux : les bureaux de cette aile doivent partir et aucune bureau ne peut venir dans cette aile"""

    #Phase 1 aile B en travaux (2)
    contraintes.append(sum( x[index(i,2,1)] * I[index(i, 2, 0)] for i in range(1, n))== 0)
    contraintes.append(sum( x[index(2,j,1)] * I[index(2, j, 0)] for j in range(1, n+1))== 1)

    #Phase 2 aile P en travaux (3)
    contraintes.append(sum( x[index(i,3,2)] * I[index(i, 3, 0)] for i in range(1, n+1))== 0)
    contraintes.append(sum( x[index(3,j,2)] * I[index(3, j, 0)] for j in range(1, n+1))== 1)

    #Phase 3 aile C/D en travaux (4)
    contraintes.append(sum( x[index(i,4,3)] * I[index(i, 4, 0)] for i in range(1, n+1))== 0)
    contraintes.append(sum( x[index(4,j,3)] * I[index(4, j, 0)] for j in range(1, n+1))== 1)


    #Phase 4 aile A en travaux
    contraintes.append(sum( x[index(i,1,4)] * I[index(i, 1, 0)] for i in range(1, n+1))== 0)
    contraintes.append(sum( x[index(1,j,4)] * I[index(1, j, 0)] for j in range(1, n+1))== 1)


     #///////////CONTRAINTES SUR u ET U ////////

    for i in range(u_dimension):
        contraintes.append(U[i,i] == 1)

    contraintes.append(u >= -1 )
    contraintes.append(u <=1 )

    contraintes.append(Y[0,0] == z)
    contraintes.append(Y[1:, 0] == u)
    contraintes.append( Y[0, 1:] == u.T)
    contraintes.append( Y[1:, 1:] == U)

    contraintes.append(z ==1)

    contraintes.append(Y >> 0)


    """Contrainte 3 : Contraintes linéaire pour les vecteurs positions """


    for i in range(1,n+1):
        for j in range(1,n):
            for k in range(1,p+1):
                contraintes.append(P[index_P(i,j,k)] ==  sum ( U[index_u(2,l,j,k-1), index_u(1,l,i,k)] for l in range(1,n+1)))


    for i in range(1,n+1):
        for j in range(1,n+1):
            for k in range(1,p+1):
                contraintes.append(u[index_u(1,i,j,k)] == 2 * x[index(i,j,k)] -1)

    for i in range(1,n+1):
        for j in range(1,n):
            for k in range(1,p+1):
                contraintes.append(u[index_u(2,i,j,k)] == 2* P[index_P(i,j,k)] -1)

    prob = cvx.Problem(function,contraintes)
    # Solving the problem
    result = prob.solve(verbose=False)

    return prob.value, P.value, x.value

val_SDP, P_SDP, x_SDP= modele_SDP_5()

print("La valeur optimale pour le probème des 5 bureaux est " + str(val_SDP))

