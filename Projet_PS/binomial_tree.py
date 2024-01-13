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
        self.p = (np.exp(r * T / n) - self.d) / (self.u - self.d) # Probabilité risque neutre
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

class BinomialTreeAM:
    def __init__(self, S0, T, r, u, K, n):
        self.S0 = S0
        self.T = T
        self.r = r
        # self.sigma = sigma
        self.K = K
        self.n = n
        # self.u = np.exp(sigma * np.sqrt(T / n))
        self.u = u
        self.d = 1 / self.u
        # avec sigma
        # self.p = (np.exp(r * T / n) - self.d) / (self.u - self.d) # Probabilité risque neutre
        self.p = (1 + self.r - self.d) / (self.u - self.d) # Probabilité risque neutre
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
        """'une option américaine peut être exercée à tout moment avant la date d'échéance 
        si son prix intrinsèque est supérieur à la valeur de continuation."""
        for i in range(self.n + 1):
            self.call_payoffs[i, self.n] = max(self.stock_tree[i, self.n] - self.K, 0)
            self.put_payoffs[i, self.n] = max(self.K - self.stock_tree[i, self.n], 0)

        for i in range(self.n - 1, -1, -1):
            self.call_payoffs[0, i] = np.exp(-self.r * self.T / self.n) * (
                    self.p * self.call_payoffs[0, i + 1] + (1 - self.p) * self.call_payoffs[1, i + 1])

            self.put_payoffs[0, i] = np.exp(-self.r * self.T / self.n) * (
                    self.p * self.put_payoffs[0, i + 1] + (1 - self.p) * self.put_payoffs[1, i + 1])

            for j in range(1, i + 1):
                intrinsic_put = max(self.K - self.stock_tree[j, i], 0)
                intrinsic_call = max(self.stock_tree[j, i] - self.K, 0)

                # Comparez la valeur intrinsèque à la valeur de continuation (H)
                self.put_payoffs[j, i] = np.exp(-self.r * self.T / self.n) * (
                        self.p * intrinsic_put + (1 - self.p) * self.put_payoffs[j + 1, i + 1])
                self.call_payoffs[j, i] = np.exp(-self.r * self.T / self.n) * (
                        self.p * intrinsic_call + (1 - self.p) * self.call_payoffs[j + 1, i + 1])

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

class ConvergenceGraph:
    def __init__(self, S0, K, r, sigma, T):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    def crr_option_price(self, N):
        dt = self.T / N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)

        stock_prices = np.zeros((N+1, N+1))
        option_prices = np.zeros((N+1, N+1))

        # Calcul des prix de l'actif sous-jacent à chaque nœud
        for i in range(N+1):
            for j in range(i+1):
                stock_prices[j, i] = self.S0 * (u ** (i-j)) * (d ** j)

        # Calcul des prix de l'option à l'échéance
        option_prices[:, N] = np.maximum(stock_prices[:, N] - self.K, 0)

        # Calcul récursif des prix de l'option du dernier nœud à l'instant initial
        for i in range(N-1, -1, -1):
            for j in range(i+1):
                option_prices[j, i] = np.exp(-self.r * dt) * (p * option_prices[j, i+1] + (1 - p) * option_prices[j+1, i+1])

        return option_prices[0, 0]

    def plot_convergence_graph(self, N_values):
        crr_prices = [self.crr_option_price(N) for N in N_values]

        # Tracé du graphique de convergence
        plt.plot(N_values, crr_prices, label='Modèle CRR')
        plt.xlabel('Nombre de périodes (N)')
        plt.ylabel('Prix de l\'option')
        plt.title('Convergence du modèle CRR')
        plt.legend()
        plt.show()