import tkinter as tk
import networkx as nx
from tkinter import ttk,  messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import decimal
import numpy as np
import matplotlib.pyplot as plt
from binomial_tree import BinomialTree
from binomial_tree import ConvergenceGraph

class BinomialTreeApp:
    """Classe principale de l'application. 
    Calcul du prix d'une option européenne à l'aide du modèle binomial de Cox-Ross-Rubinstein."""
    def __init__(self, root):
        """Fonction init, crée la fenêtre principale et les widgets."""
        self.root = root
        self.root.title("Modèle binomial | Calculateur d'options")

        # Obtenez les dimensions de l'écran
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Calculez les coordonnées x et y pour centrer la fenêtre
        x = (screen_width - 800) // 2  # Ajustez la largeur de la fenêtre selon vos besoins
        y = (screen_height - 600) // 2  # Ajustez la hauteur de la fenêtre selon vos besoins

        # Définissez la géométrie de la fenêtre
        self.root.geometry(f"300x300+{x}+{y}")

        # Configurez la couleur de fond de la fenêtre principale
        style = ttk.Style()
        style.configure("TFrame", background="#1E213D")  # Valeurs RGB (30, 33, 61)
        self.root.configure(bg="#1E213D")  # Configurez la couleur de fond de la racine

        self.create_widgets()

    def create_widgets(self):
        """Fonctions pour créer les widgets de la fenêtre."""
        # Frame principale
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Colonne gauche
        inputs_frame = ttk.Frame(main_frame, padding=(10, 10, 10, 10))
        inputs_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(inputs_frame, text="S0:").grid(row=0, column=0, sticky="w")
        self.S0_entry = ttk.Entry(inputs_frame)
        self.S0_entry.grid(row=0, column=1)

        ttk.Label(inputs_frame, text="T:").grid(row=1, column=0, sticky="w")
        self.T_entry = ttk.Entry(inputs_frame)
        self.T_entry.grid(row=1, column=1)

        ttk.Label(inputs_frame, text="r:").grid(row=2, column=0, sticky="w")
        self.r_entry = ttk.Entry(inputs_frame)
        self.r_entry.grid(row=2, column=1)

        ttk.Label(inputs_frame, text="sigma:").grid(row=3, column=0, sticky="w")
        self.sigma_entry = ttk.Entry(inputs_frame)
        self.sigma_entry.grid(row=3, column=1)

        ttk.Label(inputs_frame, text="K:").grid(row=4, column=0, sticky="w")
        self.K_entry = ttk.Entry(inputs_frame)
        self.K_entry.grid(row=4, column=1)

        ttk.Label(inputs_frame, text="n:").grid(row=5, column=0, sticky="w")
        self.n_entry = ttk.Entry(inputs_frame)
        self.n_entry.grid(row=5, column=1)

        # Colonne droite
        buttons_frame = ttk.Frame(inputs_frame)
        buttons_frame.grid(row=0, column=2, rowspan=6, padx=(10, 0), sticky="nsew")

        # Bouton pour afficher l'arbre
        ttk.Button(buttons_frame, text="Afficher l'arbre", command=self.display_tree).grid(row=0, column=0, pady=(0, 10))

        # Bouton pour calculer le Put
        ttk.Button(buttons_frame, text="Calcul : Put", command=self.calculate_put).grid(row=1, column=0, pady=5)

        # Bouton pour calculer le Call
        ttk.Button(buttons_frame, text="Calcul : Call", command=self.calculate_call).grid(row=2, column=0, pady=(5, 0))

        # Bouton pour afficher le graphique de convergence
        ttk.Button(buttons_frame, text="Afficher Convergence", command=self.display_convergence_graph).grid(row=3, column=0, pady=(5, 0))

        # Quart inférieur gauche: Arbre
        tree_frame = ttk.Frame(main_frame)
        tree_frame.grid(row=1, column=0, sticky="nsew")

        # Ajustement des poids pour que les parties de la fenêtre s'ajustent en taille
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def get_input_values(self):
        """Fonction pour obtenir les valeurs entrées par l'utilisateur."""
        try:
            S0 = float(self.S0_entry.get())
            T = float(self.T_entry.get())
            r = float(self.r_entry.get())
            sigma = float(self.sigma_entry.get())
            K = float(self.K_entry.get())
            n = int(self.n_entry.get())

            return S0, T, r, sigma, K, n

        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides.")
            return None

    def display_convergence_graph(self):
        values = self.get_input_values()
        if values is not None:
            S0, T, r, sigma, K, n = values
            convergence_graph = ConvergenceGraph(S0, K, r, sigma, T)
            N_values = np.arange(10, n + 101, 1)  # Ajustez la plage des valeurs N selon vos besoins
            convergence_graph.plot_convergence_graph(N_values)


    def display_tree(self):
        """Fonction pour afficher l'arbre."""
        values = self.get_input_values()
        if values is not None:
            S0, T, r, sigma, K, n = values
            binomial_tree = BinomialTree(S0, T, r, sigma, K, n)
            binomial_tree.build_stock_tree()
            binomial_tree.build_graph()
            binomial_tree.calculate_payoffs()
            binomial_tree.draw_tree()

    def calculate_put(self):
        """Calcul du prix du Put"""
        values = self.get_input_values()
        if values is not None:
            S0, T, r, sigma, K, n = values
            binomial_tree = BinomialTree(S0, T, r, sigma, K, n)
            binomial_tree.build_stock_tree()
            binomial_tree.calculate_payoffs()
            put_value = binomial_tree.put_payoffs[0, 0]
            messagebox.showinfo("Put (t=0)", f"La valeur du Put à l'instant 0 est : {round(put_value, 2)}")

    def calculate_call(self):
        """Calcul du prix du Call"""
        values = self.get_input_values()
        if values is not None:
            S0, T, r, sigma, K, n = values
            binomial_tree = BinomialTree(S0, T, r, sigma, K, n)
            binomial_tree.build_stock_tree()
            binomial_tree.calculate_payoffs()
            call_value = binomial_tree.call_payoffs[0, 0]
            messagebox.showinfo("Call (t=0)", f"La valeur du Call à l'instant 0 est : {round(call_value, 2)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BinomialTreeApp(root)
    root.mainloop()
