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


def adaptive_cost_encoding(G, initial_params, n_layers = 1, optimizer_method = 'COBYLA'):
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

def adaptive_cost_encoding_de(G, initial_params, n_layers = 1, optimizer_method = 'COBYLA'):
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