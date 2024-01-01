# %% IMPORTS
import decimal
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt

# %% Définir les paramètres du modèle
S0 = 20 # Prix initial de l'actif
T = 0.5 # Horizon de temps
r = 0.12 # Taux d'intérêt sans risque
sigma = 0.2 # Volatilité
K = 21 # Prix d'exercice

# Définir les paramètres de l'arbre binomial
n = 8 # Nombre de périodes
u = np.exp(sigma * np.sqrt(T / n)) # Taux de croissance de l'actif en cas de hausse
d = 1 / u # Taux de décroissance de l'actif en cas de baisse
p = (np.exp(r * T / n) - d) / (u - d) # Probabilité de hausse
print(u, d, p)

# Initialiser l'arbre binomial
stock_tree = np.zeros((n + 1, n + 1))
stock_tree[0, 0] = S0

# %% Calculer les prix de l'actif chaque noeud de l'arbre
for i in range(1, n + 1):
    stock_tree[0, i] = stock_tree[0, i - 1] * u
    for j in range(1, i + 1):
        stock_tree[j, i] = stock_tree[j - 1, i - 1] * d

# %% Créer un graphe orienté pour représenter l'arbre binomial
G = nx.DiGraph()
pos = {} # Positions des noeuds dans le graphe
node_labels = {} # Labels des noeuds

# Ajouter les noeuds et les arêtes dans le graphe
for i in range(n + 1):
    for j in range(i + 1):
        node_name = f"{i}, {j}" # Nom du noeud dans le format "i, j"
        G.add_node(node_name)
        pos[node_name] = (-j + i / 2, -i) # Positionnement inverse des noeuds
        node_labels[node_name] = f"{round(stock_tree[j, i], 2)}"

        if i < n:
            G.add_edge(node_name, f"{i + 1}, {j}") # Arête vers le noeud de la période suivante
            G.add_edge(node_name, f"{i + 1}, {j + 1}") # Arête vers le noeud de la période suivante

# %% Calcul : payoffs du call et du put à l'instant T
call_payoffs = np.zeros((n + 1, n + 1))
put_payoffs = np.zeros((n + 1, n + 1))
for i in range(n + 1):
    call_payoffs[i, n] = max(stock_tree[i, n] - K, 0)
    put_payoffs[i, n] = max(K - stock_tree[i, n], 0)

# Calcul de la valeur de l'option
for i in range(n - 1, -1, -1):
    call_payoffs[0, i] = np.exp(-r * T / n) * (p * call_payoffs[0, i + 1] + (1 - p) * call_payoffs[1, i + 1])
    put_payoffs[0, i] = np.exp(-r * T / n) * (p * put_payoffs[0, i + 1] + (1 - p) * put_payoffs[1, i + 1])
    for j in range(i + 1):
        call_payoffs[j, i] = np.exp(-r * T / n) * (p * call_payoffs[j, i + 1] + (1 - p) * call_payoffs[j + 1, i + 1])
        put_payoffs[j, i] = np.exp(-r * T / n) * (p * put_payoffs[j, i + 1] + (1 - p) * put_payoffs[j + 1, i + 1])

# Créer un dictionnaire des labels pour les noeuds avec les valeurs des payoffs
for i in range(n + 1):
    for j in range(i + 1):
        node_name = f"{i}, {j}" # Nom du noeud
        node_labels[node_name] += f"\nC: {round(call_payoffs[j, i], 2)}\nP: {round(put_payoffs[j, i], 2)}"

# %% Dessiner le graphe
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=False, node_size=1500, node_color='lightblue', edge_color='gray', arrowsize=20)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

# Dessiner l'axe centre
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax.axhline(color='black', linewidth=0)

# Décalage de l'axe centre vers le bas
plt.subplots_adjust(top=0, bottom=-0.5)

# Étiquettes des entiers positifs droite
for i in range(n + 1):
    ax.text(i - i / 2, 0.45, f'{i}', ha='center', va='center')

# Étiquettes des entiers positifs gauche
for i in range(1, n + 1):
    ax.text(-i + i / 2, 0.45, f'{i}', ha='center', va='center')

# Ajouter les noms des parties droite et gauche de l'axe
division = decimal.Decimal(n) / decimal.Decimal(3)
ax.text(division, 0.8, 'Nombre de hausses', ha='center', va='center')
ax.text(-division, 0.8, 'Nombre de baisses', ha='center', va='center')

plt.title(f'Arbre binomial pour n={n} dans le cas d\'un call et put européens')
plt.xlabel('Nombre de périodes')
plt.ylabel('Variation de prix')
plt.ylim(-n - 2.5, 1)
plt.gca().invert_yaxis() # Inverser l'axe y pour afficher l'arbre dans le bon sens
plt.axis("off")
plt.show()

# %% TEST creation fonction qui dessine l'abre
"""On va faire une fonction qui dessine l'arbre directement
1. Initialiser les paramètres du modèle : S0,T,r,sigma,k
2. Initialiser les paramètre de l'abre : n,u,d,p
3. Initialiser l'arbre : stock_tree en fonction de n et de S0 comme racine
4. Calcule des prix de l'actif dans chaque noeud de l'arbre et l'ajouter à stock_tree
5. Créer un graphe orienté G en fonction de n et de stock_tree
6. Calculer les payoffs du call et du put à l'instant T et les ajouter à stock_tree
7. Dessiner le graph G
Ainsi, cette fonction sera appelé arbre_binom en fonction de S0,T,r,sigma,k,n"""
def arbre_binom(S0,T,r,sigma,K,n):
    u = np.exp(sigma * np.sqrt(T / n)) # Taux de croissance de l'actif en cas de hausse
    d = 1 / u # Taux de décroissance de l'actif en cas de baisse
    p = (np.exp(r * T / n) - d) / (u - d) # Probabilité de hausse
    stock_tree = np.zeros((n + 1, n + 1))
    stock_tree[0, 0] = S0
    for i in range(1, n + 1):
        stock_tree[0, i] = stock_tree[0, i - 1] * u
        for j in range(1, i + 1):
            stock_tree[j, i] = stock_tree[j - 1, i - 1] * d
    G = nx.DiGraph()
    pos = {} # Positions des noeuds dans le graphe
    node_labels = {} # Labels des noeuds
    for i in range(n + 1):
        for j in range(i + 1):
            node_name = f"{i}, {j}" # Nom du noeud dans le format "i, j"
            G.add_node(node_name)
            pos[node_name] = (-j + i / 2, -i) # Positionnement inverse des noeuds
            node_labels[node_name] = f"{round(stock_tree[j, i], 2)}"
            if i < n:
                G.add_edge(node_name, f"{i + 1}, {j}") # Arête vers le noeud de la période suivante
                G.add_edge(node_name, f"{i + 1}, {j + 1}") # Arête vers le noeud de la période suivante
    call_payoffs = np.zeros((n + 1, n + 1))
    put_payoffs = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        call_payoffs[i, n] = max(stock_tree[i, n] - K, 0)
        put_payoffs[i, n] = max(K - stock_tree[i, n], 0)
    for i in range(n - 1, -1, -1):
        call_payoffs[0, i] = np.exp(-r * T / n) * (p * call_payoffs[0, i + 1] + (1 - p) * call_payoffs[1, i + 1])
        put_payoffs[0, i] = np.exp(-r * T / n) * (p * put_payoffs[0, i + 1] + (1 - p) * put_payoffs[1, i + 1])
        for j in range(i + 1):
            call_payoffs[j, i] = np.exp(-r * T / n) * (p * call_payoffs[j, i + 1] + (1 - p) * call_payoffs[j + 1, i + 1])
            put_payoffs[j, i] = np.exp(-r * T / n) * (p * put_payoffs[j, i + 1] + (1 - p) * put_payoffs[j + 1, i + 1])
    print(call_payoffs)
    for i in range(n + 1):
        for j in range(i + 1):
            node_name = f"{i}, {j}" # Nom du noeud
            node_labels[node_name] += f"\nC: {round(call_payoffs[j, i], 2)}\nP: {round(put_payoffs[j, i], 2)}"
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=False, node_size=1500, node_color='lightblue', edge_color='gray', arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.axhline(color='black', linewidth=0)
    plt.subplots_adjust(top=0, bottom=-0.5)
    division = decimal.Decimal(n) / decimal.Decimal(3)
    ax.text(division, 0.8, 'Nombre de hausses', ha='center', va='center')
    ax.text(-division, 0.8, 'Nombre de baisses', ha='center', va='center')
    plt.title(f'Arbre binomial pour n={n} dans le cas d\'un call et put européens')
    plt.xlabel('Nombre de périodes')
    plt.ylabel('Variation de prix')
    plt.ylim(-n - 2.5, 1)
    plt.gca().invert_yaxis() # Inverser l'axe y pour afficher l'arbre dans le bon sens
    plt.axis("off")
    print(u,d,p)
    plt.show()

# %% TEST
"""Test avec 
S0 = 20 # Prix initial de l'actif
T = 0.5 # Horizon de temps
r = 0.12 # Taux d'intérêt sans risque
sigma = 0.2 # Volatilité
K = 21 # Prix d'exercice
n = 3"""
arbre_binom(20,0.5,0.12,0.2,21,3)
# %%
##############################################################################################################
# %% Sous forme de classes
import decimal
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class BinomialTree:
    def __init__(self, S0, T, r, sigma, K, n):
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.K = K
        self.n = n
        self.u = np.exp(sigma * np.sqrt(T / n))
        self.d = 1 / self.u
        self.p = (np.exp(r * T / n) - self.d) / (self.u - self.d)
        self.stock_tree = np.zeros((n + 1, n + 1))
        self.call_payoffs = np.zeros((n + 1, n + 1))
        self.put_payoffs = np.zeros((n + 1, n + 1))
        self.G = nx.DiGraph()
        self.pos = {}
        self.node_labels = {}

    def build_stock_tree(self):
        self.stock_tree[0, 0] = self.S0
        for i in range(1, self.n + 1):
            self.stock_tree[0, i] = self.stock_tree[0, i - 1] * self.u
            for j in range(1, i + 1):
                self.stock_tree[j, i] = self.stock_tree[j - 1, i - 1] * self.d

    def build_graph(self):
        for i in range(self.n + 1):
            for j in range(i + 1):
                node_name = f"{i}, {j}"
                self.G.add_node(node_name)
                self.pos[node_name] = (-j + i / 2, -i)
                self.node_labels[node_name] = f"{round(self.stock_tree[j, i], 2)}"
                if i < self.n:
                    self.G.add_edge(node_name, f"{i + 1}, {j}")
                    self.G.add_edge(node_name, f"{i + 1}, {j + 1}")

    def calculate_payoffs(self):
        for i in range(self.n + 1):
            self.call_payoffs[i, self.n] = max(self.stock_tree[i, self.n] - self.K, 0)
            self.put_payoffs[i, self.n] = max(self.K - self.stock_tree[i, self.n], 0)

        for i in range(self.n - 1, -1, -1):
            self.call_payoffs[0, i] = np.exp(-self.r * self.T / self.n) * (
                    self.p * self.call_payoffs[0, i + 1] + (1 - self.p) * self.call_payoffs[1, i + 1])
            self.put_payoffs[0, i] = np.exp(-self.r * self.T / self.n) * (
                    self.p * self.put_payoffs[0, i + 1] + (1 - self.p) * self.put_payoffs[1, i + 1])
            for j in range(i + 1):
                self.call_payoffs[j, i] = np.exp(-self.r * self.T / self.n) * (
                        self.p * self.call_payoffs[j, i + 1] + (1 - self.p) * self.call_payoffs[j + 1, i + 1])
                self.put_payoffs[j, i] = np.exp(-self.r * self.T / self.n) * (
                        self.p * self.put_payoffs[j, i + 1] + (1 - self.p) * self.put_payoffs[j + 1, i + 1])

    def draw_tree(self):
        for i in range(self.n + 1):
            for j in range(i + 1):
                node_name = f"{i}, {j}"
                self.node_labels[node_name] += f"\nC: {round(self.call_payoffs[j, i], 2)}\nP: {round(self.put_payoffs[j, i], 2)}"

        plt.figure(figsize=(8, 8))
        nx.draw(self.G, self.pos, with_labels=False, node_size=1500, node_color='lightblue', edge_color='gray',
                arrowsize=20)
        nx.draw_networkx_labels(self.G, self.pos, labels=self.node_labels, font_size=10)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_visible(False)
        ax.yaxis.tick_left()
        ax.axhline(color='black', linewidth=0)
        plt.subplots_adjust(top=0, bottom=-0.5)
        division = decimal.Decimal(self.n) / decimal.Decimal(3)
        ax.text(division, 0.8, 'Nombre de hausses', ha='center', va='center')
        ax.text(-division, 0.8, 'Nombre de baisses', ha='center', va='center')
        plt.title(f'Arbre binomial pour n={self.n} dans le cas d\'un call et put européens')
        plt.xlabel('Nombre de périodes')
        plt.ylabel('Variation de prix')
        plt.ylim(-self.n - 2.5, 1)
        plt.gca().invert_yaxis()
        plt.axis("off")
        plt.show()

# %% Utilisation de la classe
arbre = BinomialTree(20, 0.5, 0.12, 0.2, 21, 3)
arbre.build_stock_tree()
arbre.build_graph()
arbre.calculate_payoffs()
arbre.draw_tree()

# %%
