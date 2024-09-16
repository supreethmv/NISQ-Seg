# Import dependencies

from qiskit.circuit.library import Diagonal
from qiskit import QuantumCircuit


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import os


from qiskit_optimization.applications import Maxcut
from qiskit_optimization.algorithms import MinimumEigenOptimizer


from qiskit.primitives import Sampler

from qiskit.result import QuasiDistribution


from scipy.optimize import minimize
import time


from qiskit_algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

from qiskit_algorithms.utils import algorithm_globals



import gurobipy as gp
from gurobipy import GRB


from deap import base, creator, tools, algorithms


import random

from qiskit import Aer, execute
import numpy as np

from qiskit.circuit import ParameterVector

from qiskit import QuantumCircuit
from math import log2, ceil

from scipy.optimize import differential_evolution

# Generate Problem Instance

def image_to_grid_graph(gray_img, sigma=0.5):
  h, w = gray_img.shape
  # Initialize graph nodes and edges
  nodes = np.zeros((h*w, 1))
  edges = []
  nx_elist = []
  # Compute node potentials and edge weights
  min_weight = 1
  max_weight = 0
  for i in range(h*w):
    x, y = i//w, i%w
    nodes[i] = gray_img[x,y]
    if x > 0:
      j = (x-1)*w + y
      weight = 1-np.exp(-((gray_img[x,y] - gray_img[x-1,y])**2) / (2 * sigma**2))
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x-1,y),np.round(weight,2)))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if y > 0:
      j = x*w + y-1
      weight = 1-np.exp(-((gray_img[x,y] - gray_img[x,y-1])**2) / (2 * sigma**2))
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x,y-1),np.round(weight,2)))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
  a=-1
  b=1
  if max_weight-min_weight:
    normalized_edges = [(node1,node2,-1*np.round(((b-a)*((edge_weight-min_weight)/(max_weight-min_weight)))+a,2)) for node1,node2,edge_weight in edges]
    normalized_nx_elist = [(node1,node2,-1*np.round(((b-a)*((edge_weight-min_weight)/(max_weight-min_weight)))+a,2)) for node1,node2,edge_weight in nx_elist]
  else:
    normalized_edges = [(node1,node2,-1*np.round(edge_weight,2)) for node1,node2,edge_weight in edges]
    normalized_nx_elist = [(node1,node2,-1*np.round(edge_weight,2)) for node1,node2,edge_weight in nx_elist]
  return nodes, edges, nx_elist, normalized_edges, normalized_nx_elist

def generate_problem_instance(height,width):
  image = np.random.rand(height,width)
  pixel_values, elist, nx_elist, normalized_elist, normalized_nx_elist = image_to_grid_graph(image)
  G = nx.grid_2d_graph(image.shape[0], image.shape[1])
  G.add_weighted_edges_from(normalized_nx_elist)
  return G, image

def generate_binary_problem_instance(height,width):
  image = np.random.rand(height,width)
  image[image < 0.5] = 0
  image[image >= 0.5] = 1
  pixel_values, elist, nx_elist, normalized_elist, normalized_nx_elist = image_to_grid_graph(image)
  G = nx.grid_2d_graph(image.shape[0], image.shape[1])
  G.add_weighted_edges_from(normalized_nx_elist)
  return G, image

def draw(G, image):
  pixel_values = image
  plt.figure(figsize=(8,8))
  default_axes = plt.axes(frameon=True)
  pos = {(x,y):(y,-x) for x,y in G.nodes()}
  nx.draw_networkx(G,
                  pos=pos,
                  node_color=1-pixel_values,
                  with_labels=True,
                  node_size=3000,
                  cmap=plt.cm.Greys,
                  alpha=0.8,
                  ax=default_axes)
  nodes = nx.draw_networkx_nodes(G, pos, node_color=1-pixel_values,
                  node_size=3000,
                  cmap=plt.cm.Greys)
  nodes.set_edgecolor('k')
  edge_labels = nx.get_edge_attributes(G, "weight")
  nx.draw_networkx_edge_labels(G,
                              pos=pos,
                             edge_labels=edge_labels)
  

  # Parametric Gate Encoding (PGE)

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

# Experiments for PGE

heights = [2,4,8]

scipy_optimizer_methods = ["COBYLA", "Powell", "CG", "BFGS", "L-BFGS-B", "SLSQP", "Genetic"]

initial_params_seed = 123

optimization_iteration_count = 0

base_path = 'results/'

seeds = [111, 222,333,444,555]



for seed in seeds:
  report_filename = base_path + 'ParameterEncoding_' +  str(seed) + '.txt'
  for height in heights:
    width = height
    print(f'height: {height}, width: {width}, n: {height*width}')
    np.random.seed(seed=seed)
    G, image = generate_problem_instance(height, width)
    print("Image Generated: ",image)
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.show()
  
    for optimizer_method in scipy_optimizer_methods:
      print(f"Executing QC with {optimizer_method} optimizer for {height}*{height} image.")
      try:
        start_time = time.time()
        success_flag, new_nisq_solution, new_nisq_expectation_value, new_nisq_cut_value = new_nisq_algo_solver(G, 
                                                                                                               optimizer_method = optimizer_method, 
                                                                                                               initial_params_seed=initial_params_seed)
        new_nisq_tte = (time.time() - start_time)
        print(new_nisq_solution, new_nisq_expectation_value, new_nisq_cut_value)
      except:
        print(f"Execution Failed for {optimizer_method} optimizer for {height}*{height} image.")
        continue
      print("New NISQ done for",optimizer_method,"optimizer with a success status :", success_flag)
      print(f"Appending the results of {height}*{height} image using QC with and {optimizer_method} optimizer.")
      row = []
      row.append(int(G.number_of_nodes()))
      row.append(int(height))
      row.append(int(width))
      row.append(success_flag)
      row.append(''.join(map(str, map(int, (new_nisq_solution)))))
      row.append(np.round(new_nisq_tte,6))
      row.append(np.round(new_nisq_cut_value,4))
      row.append(np.round(new_nisq_expectation_value,4))
      row.append(optimization_iteration_count)
      row.append(optimizer_method)
      report_file_obj = open(os.path.join(report_filename),'a+')
      report_file_obj.write('__'.join(map(str,row))+'\n')
      report_file_obj.close()
    print('\n')


# Ancilla Basis Encoding (ABE)

def get_circuit(nq, n_layers):
  theta = ParameterVector('θ', length=nq*n_layers)
  qc = QuantumCircuit(nq)
  qc.h(range(nq))

  for layer_i in range(n_layers):
    # if layer_i%2:
    for qubit_i in range(nq - 1):
      qc.cx(qubit_i, (qubit_i + 1)%nq)
    for qubit_i in range(nq):
      qc.ry(theta[(nq*layer_i)+qubit_i], qubit_i)
  return qc


def get_projectors(probabilities, n_c):
  P = [0] * n_c
  P1 = [0] * n_c
  for k, v in probabilities.items():
    index = int(k[1:], 2)
    P[index] += v
    if k[0] == '1':
      P1[index] += v
  return P, P1

def objective_value(x: np.ndarray, w: np.ndarray) -> float:
    cost = 0
    for i in range(len(x)):
        for j in range(len(x)):
            cost = cost + w[i,j]*x[i]*(1-x[j])
    return cost

def evaluate_cost(params, circuit, qubo_matrix):
  global ultimate_valid_probabilities
  global penultimate_valid_probabilities
  global optimization_iteration_count
  global n_shots
  optimization_iteration_count += 1
  # Create a copy of the circuit to avoid modifying the original
  working_circuit = circuit.copy()
  # Apply measurements
  working_circuit.measure_all()
  # Bind the parameters to the circuit
  bound_circuit = working_circuit.assign_parameters(params)
  # Run the circuit on a simulator
  simulator = Aer.get_backend('qasm_simulator')
  job = execute(bound_circuit, simulator, shots=n_shots)
  result = job.result()
  counts = result.get_counts()
  # Compute the probabilities of the states
  probabilities = {state: counts[state] / n_shots for state in counts}
  n_c = 2**(len(list(probabilities.keys())[0])-1)
  P, P1 = get_projectors(probabilities, n_c)
  if 0 in P:
    return 100000
  else:
    # Update the last valid probabilities
    penultimate_valid_probabilities = ultimate_valid_probabilities.copy()
    ultimate_valid_probabilities = probabilities.copy()
  cost = 0
  for i in range(len(qubo_matrix)):
    for j in range(len(qubo_matrix)):
      if i==j:
        cost += qubo_matrix[i][j] * P1[i]/P[i]
      else:
        cost += (qubo_matrix[i][j]*P1[i]*P1[j])/(P[i]*P[j])
  # print('min_cut_cost :', min_cut_cost)
  print(f'@ Iteration {optimization_iteration_count} Cost :',cost)
  return cost

def get_final_measurement_binary_string(circuit, params):
  working_circuit = circuit.copy()
  working_circuit.measure_all()
  bound_circuit = working_circuit.assign_parameters(params)
  simulator = Aer.get_backend('qasm_simulator')
  job = execute(bound_circuit, simulator, shots=1024)
  result = job.result()
  counts = result.get_counts()
  probabilities = {state: counts[state] / 1024 for state in counts}
  return probabilities

def decode_probabilities(probabilities):
  binary_solution = []
  n_r = len(list(probabilities.keys())[0])-1
  n_c = 2**(n_r)
  for i in range(n_c):
    if '0' + format(i, 'b').zfill(n_r) in probabilities and '1' + format(i, 'b').zfill(n_r) in probabilities:
      if probabilities['0' + format(i, 'b').zfill(n_r)] > probabilities['1' + format(i, 'b').zfill(n_r)]:
        binary_solution.append(0)
      else:
        binary_solution.append(1)
    elif '0' + format(i, 'b').zfill(n_r) in probabilities:
      binary_solution.append(0)
    elif '1' + format(i, 'b').zfill(n_r) in probabilities:
      binary_solution.append(1)
  return binary_solution

def decode_binary_string(x, height, width):
  mask = np.zeros([height, width])
  for index,segment in enumerate(x):
    mask[index//width,index%width]=segment
  return mask


def ancilla_basis_encoding(G, initial_params, n_layers = 1, optimizer_method = 'COBYLA'):
  w = nx.adjacency_matrix(G).todense()
  max_cut = Maxcut(-w)
  qp = max_cut.to_quadratic_program()
  linear = qp.objective.linear.coefficients.toarray(order=None, out=None)
  quadratic = qp.objective.quadratic.coefficients.toarray(order=None, out=None)
  linear = {int(idx):-round(value,2) for idx,value in enumerate(linear[0])}
  quadratic = {(int(iy),int(ix)):-quadratic[iy, ix] for iy, ix in np.ndindex(quadratic.shape) if iy<ix and abs(quadratic[iy, ix])!=0}
  nc = len(linear)
  nr = ceil(log2(nc))
  nq = nr + 1
  qc = get_circuit(nq, n_layers)
  global ultimate_valid_probabilities
  global penultimate_valid_probabilities
  global optimization_iteration_count
  ultimate_valid_probabilities = []
  penultimate_valid_probabilities = []
  optimization_iteration_count = 0
  options = {}
  optimization_result = minimize(
      fun=evaluate_cost,
      x0=initial_params,
      args=(qc, w),
      method=optimizer_method,
      bounds=tuple([(-np.pi, np.pi) for _ in range(len(initial_params))]),
      options=options)
  if not len(ultimate_valid_probabilities):
    return False, [0]*nc, 0, 0, [0]*initial_params
  binary_string_solution = decode_probabilities(ultimate_valid_probabilities)
  minimization_cost = optimization_result.fun
  optimal_params = optimization_result.x
  cut_cost = 0
  cut_cost = objective_value(np.array(list(map(int,binary_string_solution))), w)
  return optimization_result.success, binary_string_solution, minimization_cost, cut_cost, optimal_params

def ancilla_basis_encoding_de(G, initial_params, n_layers = 1):
  w = nx.adjacency_matrix(G).todense()
  max_cut = Maxcut(-w)
  qp = max_cut.to_quadratic_program()
  linear = qp.objective.linear.coefficients.toarray(order=None, out=None)
  quadratic = qp.objective.quadratic.coefficients.toarray(order=None, out=None)
  linear = {int(idx):-round(value,2) for idx,value in enumerate(linear[0])}
  quadratic = {(int(iy),int(ix)):-quadratic[iy, ix] for iy, ix in np.ndindex(quadratic.shape) if iy<ix and abs(quadratic[iy, ix])!=0}
  nc = len(linear)
  nr = ceil(log2(nc))
  nq = nr + 1
  qc = get_circuit(nq, n_layers)
  global ultimate_valid_probabilities
  global penultimate_valid_probabilities
  global optimization_iteration_count
  ultimate_valid_probabilities = []
  penultimate_valid_probabilities = []
  optimization_iteration_count = 0
  
  optimization_result = differential_evolution(
      func=evaluate_cost,
      args = (qc, w),
      bounds=tuple([(-np.pi, np.pi) for _ in range(len(initial_params))])
      )
  if not len(ultimate_valid_probabilities):
    return False, [0]*nc, 0, 0, [0]*initial_params
  binary_string_solution = decode_probabilities(ultimate_valid_probabilities)
  minimization_cost = optimization_result.fun
  optimal_params = optimization_result.x
  cut_cost = 0
  cut_cost = objective_value(np.array(list(map(int,binary_string_solution))), w)
  return optimization_result.success, binary_string_solution, minimization_cost, cut_cost, optimal_params


# Experiments for Minimal Encoding

n_shots = 65536

ultimate_valid_probabilities = []
penultimate_valid_probabilities = []

optimization_iteration_count = 0

base_path = './'

start_layers = 1
max_layers = 5
initial_params_seed = 123
scipy_optimizer_methods = ["COBYLA", "Powell", "CG", "BFGS", "L-BFGS-B", "SLSQP"]


heights = [2,4]

seeds = [111,222,333,444,555]

for seed in seeds:
  report_filename = base_path + 'AncillaBasisStateEncoding_' +  str(seed) + '_' + str(n_shots) + '.txt'
  for height in heights:
    width = height
    print(f'height: {height}, width: {width}, n: {height*width}')
    # G, image = generate_binary_problem_instance(height, width)
    np.random.seed(seed=seed)
    G, image = generate_problem_instance(height, width)
    print("Image Generated: ",image)
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.show()

    for scipy_optimizer_method in scipy_optimizer_methods:
      print("Maximum number of layers :", max_layers)
      for n_layers in range(start_layers, max_layers+1,1):
        nc = len(G.nodes())
        nr = ceil(log2(nc))
        nq = nr + 1
        initial_params = np.random.uniform(low=-np.pi, high=np.pi, size=nq*n_layers)
        print(f"Executing QC with {n_layers} layers and {scipy_optimizer_method} optimizer for {height}*{height} image.")
        # try:
        start_time = time.time()
        success_flag, minimal_encoding_solution, minimal_encoding_value, minimal_encoding_cut_value, optimal_params = ancilla_basis_encoding_de(G,
                                                                                                                      initial_params,
                                                                                                                      n_layers = n_layers,
                                                                                                                      optimizer_method = scipy_optimizer_method)
        minimal_encoding_tte = (time.time() - start_time)
        print("New NISQ done for",scipy_optimizer_method,"optimizer with a success status :", success_flag)
        print(f"Appending the results of {height}*{height} image using QC with {n_layers} layers and {scipy_optimizer_method} optimizer.")
        row = []
        row.append(int(G.number_of_nodes()))
        row.append(int(height))
        row.append(int(width))
        row.append(success_flag)
        row.append(''.join(map(str, map(int, (minimal_encoding_solution)))))
        row.append(np.round(minimal_encoding_tte,6))
        row.append(n_layers)
        row.append(np.round(minimal_encoding_cut_value,4))
        row.append(np.round(minimal_encoding_value,4))
        row.append(optimization_iteration_count)
        row.append(scipy_optimizer_method)
        # row.append(''.join(map(str, map(float, (optimal_params)))))
        report_file_obj = open(os.path.join(report_filename),'a+')
        report_file_obj.write('__'.join(map(str,row))+'\n')
        report_file_obj.close()
    print('\n')



# Adaptive Cost Encoding (ACE)

def evaluate_mincut_cost(params, circuit, w):
  global optimization_iteration_count
  global n_shots
  global best_cost
  global best_cost_binary_solution
  optimization_iteration_count += 1
  working_circuit = circuit.copy()

  # Apply measurements
  working_circuit.measure_all()

  # Bind the parameters to the circuit
  bound_circuit = working_circuit.assign_parameters(params)

  # Run the circuit on a simulator
  simulator = Aer.get_backend('qasm_simulator')
  job = execute(bound_circuit, simulator, shots=n_shots)
  result = job.result()
  counts = result.get_counts()

  probabilities = {state: counts[state] / n_shots for state in counts}
  n_c = 2**(len(list(probabilities.keys())[0])-1)
  P, P1 = get_projectors(probabilities, n_c)

  if 0 in P:
    return 100000

  binary_solution = decode_probabilities(probabilities)
  cost = objective_value(binary_solution, w)
  if cost < best_cost:
    best_cost = cost
    best_cost_binary_solution = binary_solution

  print(f'@ Iteration {optimization_iteration_count} Cost :',np.round(cost,2))
  print(f'Best Cost Found:',np.round(best_cost,2),"\n")
  return cost


def adaptive_cost_encoding(G, initial_params, n_layers = 1, max_iter = 2000, optimizer_method = 'COBYLA'):
  w = nx.adjacency_matrix(G).todense()
  max_cut = Maxcut(-w)
  qp = max_cut.to_quadratic_program()
  linear = qp.objective.linear.coefficients.toarray(order=None, out=None)
  quadratic = qp.objective.quadratic.coefficients.toarray(order=None, out=None)
  linear = {int(idx):-round(value,2) for idx,value in enumerate(linear[0])}
  quadratic = {(int(iy),int(ix)):-quadratic[iy, ix] for iy, ix in np.ndindex(quadratic.shape) if iy<ix and abs(quadratic[iy, ix])!=0}
  nc = len(linear)
  nr = ceil(log2(nc))
  nq = nr + 1
  qc = get_circuit(nq, n_layers)
  global optimization_iteration_count
  global best_cost
  global best_cost_binary_solution

  optimization_iteration_count = 0
  options = {}
  optimization_result = minimize(
      fun=evaluate_mincut_cost,
      x0=initial_params,
      args=(qc, w),
      method=optimizer_method,
      options=options)

  minimization_cost = optimization_result.fun
  return optimization_result.success, best_cost_binary_solution, minimization_cost, best_cost

def adaptive_cost_encoding_de(G, initial_params, n_layers = 1, max_iter = 200, optimizer_method = 'COBYLA'):
  w = nx.adjacency_matrix(G).todense()
  max_cut = Maxcut(-w)
  qp = max_cut.to_quadratic_program()
  linear = qp.objective.linear.coefficients.toarray(order=None, out=None)
  quadratic = qp.objective.quadratic.coefficients.toarray(order=None, out=None)
  linear = {int(idx):-round(value,2) for idx,value in enumerate(linear[0])}
  quadratic = {(int(iy),int(ix)):-quadratic[iy, ix] for iy, ix in np.ndindex(quadratic.shape) if iy<ix and abs(quadratic[iy, ix])!=0}
  nc = len(linear)
  nr = ceil(log2(nc))
  nq = nr + 1
  qc = get_circuit(nq, n_layers)
  global best_cost
  global best_cost_binary_solution
  optimization_iteration_count = 0
  
  optimization_result = differential_evolution(
      func=evaluate_mincut_cost,
      args = (qc, w),
      bounds=tuple([(-np.pi, np.pi) for _ in range(len(initial_params))]),
      )
  minimization_cost = optimization_result.fun
  return optimization_result.success, best_cost_binary_solution, minimization_cost, best_cost


# Experiments for Minimal Encoding

base_path = './'

n_shots = 65536

optimization_max_iter = 10000
optimization_iteration_count = 0

start_layers = 1
max_layers = 1
initial_params_seed = 123
scipy_optimizer_methods = ["COBYLA", "Powell", "CG", "BFGS", "L-BFGS-B", "SLSQP"]
# scipy_optimizer_methods = ["Differential_Evolution"]


heights = [2,4,8]

seeds = [111,222,333,444,555]


for seed in seeds:
  report_filename = base_path + 'AncillaBasisStateEncoding_newCost_' +  str(seed) + '_' + str(n_shots) + '.txt'
  for height in heights:
    width = height
    print(f'height: {height}, width: {width}, n: {height*width}')
    np.random.seed(seed=seed)
    G, image = generate_problem_instance(height, width)
    print("Image Generated: ",image)
    for scipy_optimizer_method in scipy_optimizer_methods:
      print("Maximum number of layers :", max_layers)
      for n_layers in range(start_layers, max_layers+1,1):
        best_cost = 1000000
        best_cost_binary_solution = [0] * (height*width)
        nc = len(G.nodes())
        nr = ceil(log2(nc))
        nq = nr + 1
        initial_params = np.random.uniform(low=-np.pi, high=np.pi, size=nq*n_layers)
        print(f"Executing QC with {n_layers} layers and {scipy_optimizer_method} optimizer for {height}*{height} image.")
        # try:
        start_time = time.time()
        success_flag, minimal_encoding_solution, minimal_encoding_value, minimal_encoding_cut_value = adaptive_cost_encoding_de(G,
                                                                                                                         initial_params,
                                                                                                                         n_layers = n_layers,
                                                                                                                         max_iter = optimization_max_iter,
                                                                                                                         optimizer_method = scipy_optimizer_method)
        minimal_encoding_tte = (time.time() - start_time)
        print("New NISQ done for",scipy_optimizer_method,"optimizer with a success status :", success_flag)
        print(f"Appending the results of {height}*{height} image using QC with {n_layers} layers and {scipy_optimizer_method} optimizer.")
        row = []
        row.append(int(G.number_of_nodes()))
        row.append(int(height))
        row.append(int(width))
        row.append(success_flag)
        row.append(''.join(map(str, map(int, (minimal_encoding_solution)))))
        row.append(np.round(minimal_encoding_tte,6))
        row.append(n_layers)
        row.append(np.round(minimal_encoding_cut_value,4))
        row.append(np.round(minimal_encoding_value,4))
        row.append(optimization_iteration_count)
        row.append(scipy_optimizer_method)
        report_file_obj = open(os.path.join(report_filename),'a+')
        report_file_obj.write('__'.join(map(str,row))+'\n')
        report_file_obj.close()
    print('\n')