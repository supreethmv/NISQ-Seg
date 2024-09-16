import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import networkx as nx
from deap import base, creator, tools, algorithms
import random
from scipy.optimize import minimize


def R(theta):
  if abs(theta) > 2*np.pi or abs(theta) < 0:
    theta = abs(theta) - (np.floor(abs(theta)/(2*np.pi))*(2*np.pi))
  return 0 if 0 <= theta < np.pi else 1

def cost_fn(params, hermitian_observable):
  global optimization_iteration_count
  optimization_iteration_count += 1
  N = int(np.log2(len(params)))
  circuit_psi = QuantumCircuit(N)
  for i in range(N):
    circuit_psi.h(i)
  diagonal_elements = [np.exp(1j * np.pi * R(param)) for param in params]
  diag_gate = Diagonal(diagonal_elements)
  circuit_psi.append(diag_gate, range(N))
  op_observable = SparsePauliOp.from_operator(hermitian_observable)
  cost = Estimator().run(circuit_psi, op_observable).result().values[0]
  print(f'@ Iteration {optimization_iteration_count} Cost :',cost)
  return cost

def decode(optimal_params):
  return [R(param) for param in optimal_params]


def objective_value(x: np.ndarray, w: np.ndarray) -> float:
    cost = 0
    for i in range(len(x)):
        for j in range(len(x)):
            cost = cost + w[i,j]*x[i]*(1-x[j])
    return cost

def new_nisq_ga_solver2(G, population_size=50, crossover_probability=0.7, mutation_probability=0.2, number_of_generations=50):
    n = G.number_of_nodes()
    w = nx.adjacency_matrix(G).todense()
    D_G = np.diag(list(dict(G.degree()).values()))
    A_G = w
    L_G = D_G - A_G
    n_padding = (2**int(np.ceil(np.log2(n))) - n)
    L_G = np.pad(L_G, [(0, n_padding), (0, n_padding)], mode='constant')
    H = L_G

    # Genetic Algorithm Setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.5, 2*np.pi)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluate function with padding
    def evaluate_with_padding(individual):
        padded_individual = individual + [0] * n_padding  # Append constant values for padding
        return cost_fn(padded_individual, H),

    toolbox.register("evaluate", evaluate_with_padding)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run Genetic Algorithm
    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability, ngen=number_of_generations, verbose=False)

    # Extract the best solution
    best_ind = tools.selBest(population, 1)[0]
    optimal_params = best_ind + [0] * n_padding  # Append constant values for padding
    expectation_value = best_ind.fitness.values[0]

    # Decode and calculate cut value
    x = np.real(decode(optimal_params))
    x = x[:n]
    cut_value = objective_value(x, w)

    return x, expectation_value, cut_value


def new_nisq_algo_solver(G, optimizer_method = 'Genetic', initial_params_seed=123):
  global optimization_iteration_count
  optimization_iteration_count = 0
  if optimizer_method == 'Genetic':
    x, expectation_value, cut_value = new_nisq_ga_solver2(G,
                                                         population_size = 50,
                                                         crossover_probability = 0.7,
                                                         mutation_probability = 0.2,
                                                         number_of_generations = 50)
    success_flag = True
  else:
    n = G.number_of_nodes()
    w = nx.adjacency_matrix(G).todense()
    D_G = np.diag(list(dict(G.degree()).values()))
    A_G = w
    L_G = D_G - A_G
    n_padding = (2**int(np.ceil(np.log2(n)))-n)
    L_G = np.pad(L_G, [(0, n_padding), (0, n_padding) ], mode='constant')
    H = L_G
    np.random.seed(seed=initial_params_seed)
    initial_params = np.random.uniform(low=0.5, high=2*np.pi , size=(n+n_padding))

    options = {}
    result = minimize(
        fun=cost_fn,
        x0=initial_params,
        args=H,
        method=optimizer_method,
        options=options)
    
    optimal_params, expectation_value = result.x, result.fun
    x = np.real(decode(optimal_params))
    x = x[:n]
    cut_value = objective_value(x, w)
    success_flag = result.success
  return success_flag, x, expectation_value, cut_value