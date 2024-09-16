import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, ParameterVector
import networkx as nx
from qiskit_optimization.applications import Maxcut
from math import log2, ceil
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


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