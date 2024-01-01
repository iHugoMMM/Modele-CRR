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
