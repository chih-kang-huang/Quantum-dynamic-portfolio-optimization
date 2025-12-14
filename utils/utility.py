"""
===========================================================================
 Utility functions for Portfolio Optimization with QUBO
===========================================================================
 Authors: Ali Abbassi, Jui-Ting Lu, Chih-Kang Huang
 Date: 2025-09-30
 Description:
     This package contains all the utility functions required for our demonstration.
===========================================================================
"""
import numpy as np
import pandas as pd
import yfinance as yf
import dimod
from dimod import ExactSolver
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

def decode(bits, N=2, Nq=1, T=2):

    stride = Nq + 1                  # bits per variable
    block_size = N * stride          # bits per timestep
    Delta = 1.0 / (2**(stride) - 1)  # scaling factor
    
    decoded = np.zeros((T, N))
    
    for t in range(T):
        base = t * block_size
        for i in range(N):
            seg = bits[base + i*stride : base + (i+1)*stride]
            decoded[t, i] = Delta * sum((1 << q) * seg[q] for q in range(stride))
    
    return decoded


def decode_solution(sample, N=2, Nq=3, T=2):
    Delta = 1.0/(2**(Nq+1)-1)  # 1/15
    x = np.zeros((T,N))
    stride = Nq+1  # 4
    for (label, t, i, q), bit in sample.items():
        if bit == 1:
            x[t][i] += Delta * (2**q)
    return x

def build_qubo_dynamic_mv_bqm(
    mu0, mu1, Sigma0, Sigma1,
    c0, c1, lam, eta0, A0, A1,
    Nq
):
    """
    Build the QUBO/BQM for H=1 dynamic mean-variance with transaction and budget constraints.
    """
    # ---------- basic checks ----------
    mu0 = np.asarray(mu0, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    Sigma0 = np.asarray(Sigma0, dtype=float)
    Sigma1 = np.asarray(Sigma1, dtype=float)

    N = mu0.shape[0]
    assert mu1.shape[0] == N
    assert Sigma0.shape == (N, N) and Sigma1.shape == (N, N)
    assert np.allclose(Sigma0, Sigma0.T), "Sigma0 must be symmetric"
    assert np.allclose(Sigma1, Sigma1.T), "Sigma1 must be symmetric"

    # ---------- encoding ----------
    pow2 = 2.0 ** np.arange(Nq + 1)
    Delta = 1.0 / (2.0 ** (Nq + 1) - 1.0)
    order = [('x', t, i, q) for t in (0, 1) for i in range(N) for q in range(Nq + 1)]
    index_of = {v: k for k, v in enumerate(order)}
    num_vars = len(order)

    # ---------- helpers ----------
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, vartype=dimod.BINARY)
    for v in order:
        bqm.add_variable(v, 0.0)

    def add_qubo_term(u, v, coeff):
        """Accumulate coeff * b_u * b_v into the BQM without double counting."""
        if abs(coeff) < 1e-12:
            return
        if u == v:
            bqm.add_linear(u, coeff)     # accumulate linear
        else:
            if v < u:                    # enforce canonical order
                u, v = v, u
            bqm.add_interaction(u, v, coeff)

    offset = 0.0

    # ---------- build terms ----------
    # 1) Mean-variance at t=0,1
    for t, (c_t, Sigma_t, mu_t) in enumerate(((c0, Sigma0, mu0), (c1, Sigma1, mu1))):
        for i in range(N):
            for j in range(N):
                s_ij = Sigma_t[i, j]
                if s_ij == 0.0: continue
                for q in range(Nq + 1):
                    for r in range(Nq + 1):
                        coeff = c_t * (Delta**2) * pow2[q] * pow2[r] * s_ij
                        add_qubo_term(('x', t, i, q), ('x', t, j, r), coeff)
        for i in range(N):
            for q in range(Nq + 1):
                coeff = -c_t * lam * Delta * pow2[q] * mu_t[i]
                add_qubo_term(('x', t, i, q), ('x', t, i, q), coeff)

    # 2) Transaction term
    for i in range(N):
        for q in range(Nq + 1):
            for r in range(Nq + 1):
                coeff = eta0 * (Delta**2) * pow2[q] * pow2[r]
                add_qubo_term(('x', 1, i, q), ('x', 1, i, r), coeff)
                add_qubo_term(('x', 0, i, q), ('x', 0, i, r), coeff)
                coeff_cross = -2.0 * eta0 * (Delta**2) * pow2[q] * pow2[r]
                add_qubo_term(('x', 1, i, q), ('x', 0, i, r), coeff_cross)

    # 3) Budget constraints
    for t, A_t in enumerate((A0, A1)):
        for i in range(N):
            for j in range(N):
                for q in range(Nq + 1):
                    for r in range(Nq + 1):
                        coeff = A_t * (Delta**2) * pow2[q] * pow2[r]
                        add_qubo_term(('x', t, i, q), ('x', t, j, r), coeff)
        for i in range(N):
            for q in range(Nq + 1):
                coeff = -2.0 * A_t * Delta * pow2[q]
                add_qubo_term(('x', t, i, q), ('x', t, i, q), coeff)
        offset += A_t

    # ---------- matrix form ----------
    H = bqm.to_numpy_matrix(variable_order=order)
    H = (H + H.T) / 2.0  # ensure symmetry

    return bqm, H, order, offset

def build_qubo_dynamic_mv(
    mu_list, Sigma_list, c_list, lam, eta_list, A_list, Nq
):
    """
    Build the QUBO/BQM for H=1 dynamic mean-variance with transaction and budget constraints.
    """
    # ---------- basic checks ----------
    T = len(mu_list)
    N = mu_list[0].shape[0]
    for mu in mu_list:
        mu = np.asarray(mu, dtype=float)
        assert mu.shape[0] == N
    for Sigma in Sigma_list:
        Sigma = np.asarray(Sigma, dtype=float)
        assert Sigma.shape == (N, N)
        assert np.allclose(Sigma, Sigma.T) #Sigma0 must be symmetric

    # ---------- encoding ----------
    pow2 = 2.0 ** np.arange(Nq + 1)
    Delta = 1.0 / (2.0 ** (Nq + 1) - 1.0)
    order = [('x', t, i, q) for t in range(T) for i in range(N) for q in range(Nq + 1)]
    index_of = {v: k for k, v in enumerate(order)}
    num_vars = len(order)

    # ---------- helpers ----------
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, vartype=dimod.BINARY)
    for v in order:
        bqm.add_variable(v, 0.0)

    def add_qubo_term(u, v, coeff):
        """Accumulate coeff * b_u * b_v into the BQM without double counting."""
        if abs(coeff) < 1e-12:
            return
        if u == v:
            bqm.add_linear(u, coeff)     # accumulate linear
        else:
            if v < u:                    # enforce canonical order
                u, v = v, u
            bqm.add_interaction(u, v, coeff)

    offset = 0.0

    # ---------- build terms ----------
    # 1) Mean-variance at t=0,1,...,H-1
    for t in range(T):
        c_t = c_list[t]
        Sigma_t = Sigma_list[t]
        mu_t = mu_list[t]
        for i in range(N):
            for j in range(N):
                s_ij = Sigma_t[i, j]
                if s_ij == 0.0: continue
                for q in range(Nq + 1):
                    for r in range(Nq + 1):
                        coeff = c_t * (Delta**2) * pow2[q] * pow2[r] * s_ij
                        add_qubo_term(('x', t, i, q), ('x', t, j, r), coeff)
        for i in range(N):
            for q in range(Nq + 1):
                coeff = -c_t * lam * Delta * pow2[q] * mu_t[i]
                add_qubo_term(('x', t, i, q), ('x', t, i, q), coeff)

    # 2) Transaction term
    for t in range(T-1):
        eta_t = eta_list[t]
        for i in range(N):
            for q in range(Nq + 1):
                for r in range(Nq + 1):
                    coeff = eta_t * (Delta**2) * pow2[q] * pow2[r]
                    add_qubo_term(('x', t+1, i, q), ('x', t+1, i, r), coeff)
                    add_qubo_term(('x', t, i, q), ('x', t, i, r), coeff)
                    coeff_cross = -2.0 * eta_t * (Delta**2) * pow2[q] * pow2[r]
                    add_qubo_term(('x', t+1, i, q), ('x', t, i, r), coeff_cross)

    # 3) Budget constraints
    for t in range(T):
        A_t = A_list[t]        
        for i in range(N):
            for j in range(N):
                for q in range(Nq + 1):
                    for r in range(Nq + 1):
                        coeff = A_t * (Delta**2) * pow2[q] * pow2[r]
                        add_qubo_term(('x', t, i, q), ('x', t, j, r), coeff)
        for i in range(N):
            for q in range(Nq + 1):
                coeff = -2.0 * A_t * Delta * pow2[q]
                add_qubo_term(('x', t, i, q), ('x', t, i, q), coeff)
        offset += A_t

    # ---------- matrix form ----------
    H = bqm.to_numpy_matrix(variable_order=order)
    H = (H + H.T) / 2.0  # ensure symmetry

    return bqm, H, order, offset

def build_qubo_dynamic_from_portfolio_model(portfolio_model):
    """
    Build the QUBO/BQM for dynamic mean-variance with transaction and budget constraints.
    
    Parameters
    ----------
    portfolio_model : PortfolioModel
    """

    return build_qubo_dynamic_mv(
        portfolio_model.mu_list, 
        portfolio_model.Sigma_list, 
        portfolio_model.c_list, 
        portfolio_model.lam, 
        portfolio_model.eta_list, 
        portfolio_model.A_list, 
        portfolio_model.Nq )

             
def ising_to_pauli(linear, quadratic, offset, index_of):
    n_qubits = len(index_of)
    terms = []
    coeffs = []

    # Linear terms
    for var, h in linear.items():
        idx = index_of[var]  # map variable → qubit index
        z = ['I'] * n_qubits
        z[idx] = 'Z'
        terms.append(''.join(reversed(z)))
        coeffs.append(h)

    # Quadratic terms
    for (u, v), J in quadratic.items():
        i = index_of[u]
        j = index_of[v]
        z = ['I'] * n_qubits
        z[i] = 'Z'
        z[j] = 'Z'
        terms.append(''.join(reversed(z)))
        coeffs.append(J)

    return SparsePauliOp.from_list(list(zip(terms, coeffs))), offset


def process_sampleset(sampleset, order, mu_list, Nq, offset=0, decode_func=None):
    """
    Convert a D-Wave SampleSet into a processed DataFrame with decoded variables and energy metrics.

    Parameters
    ----------
    sampleset : dimod.SampleSet
        The result from a quantum annealer (or classical sampler).
    order : list
        List of variable names specifying the bitstring order.
    mu_list : list of arrays
        List of mu arrays used in decoding.
    Nq : int
        Number of qubits per variable.
    offset : float, optional
        Energy offset to adjust raw energies (default is 0).
    decode_func : function
        Function to decode bitstrings into continuous variables. Signature: decode_func(bits, N, Nq, T).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bitstring, energy_raw, energy_shifted, occurrences,
        x0, x1, energy_percentage, x0_diff, x1_diff, sorted by energy_shifted.
    """
    rows = []

    for sample, energy, occ in sampleset.data(['sample', 'energy', 'num_occurrences']):
        bits = [sample[v] for v in order]
        x0, x1 = decode_func(bits, N=len(mu_list[0]), Nq=Nq, T=len(mu_list))
        rows.append({
            "bitstring": bits,
            "energy_raw": energy,
            "energy_shifted": energy + offset,
            "occurrences": occ,
            "x0": np.array(x0),
            "x1": np.array(x1)
        })

    df = pd.DataFrame(rows).sort_values("energy_shifted").reset_index(drop=True)
    
    best_energy = df['energy_shifted'][0]
    best_weight = (df['x0'][0], df['x1'][0])
    
    df['energy_percentage'] = np.round((df['energy_shifted'] - best_energy)/abs(best_energy) * 100, 1)
    df['x0_diff'] = df['x0'].apply(lambda x: x - best_weight[0])
    df['x1_diff'] = df['x1'].apply(lambda x: x - best_weight[1])
    
    return df


def solve_continuous_portfolio_longonly(Sigma0, Sigma1, mu0, mu1,
                                        c0=1.0, c1=0.6, lamb=0.5, eta0=10.0,
                                        x0_init=None, x1_init=None):
    """
    Solve the continuous portfolio optimization with non-negativity constraints:
    
    minimize E(x0,x1) =
        c0 * (x0^T Sigma0 x0 - lambda * mu0^T x0)
      + c1 * (x1^T Sigma1 x1 - lambda * mu1^T x1)
      + eta0 * ||x1 - x0||^2
    
    subject to sum(x0) = 1, sum(x1) = 1, and x0 >= 0, x1 >= 0.
    
    Parameters
    ----------
    Sigma0, Sigma1 : ndarray (N,N)
        Covariance matrices.
    mu0, mu1 : ndarray (N,)
        Expected returns.
    c0, c1 : float
        Time weights.
    lamb : float
        Risk-aversion.
    eta0 : float
        Rebalancing penalty.
    x0_init, x1_init : ndarray (optional)
        Initial guesses for x0 and x1.
    
    Returns
    -------
    result : dict
        {
          "x0": optimal allocation at t=0,
          "x1": optimal allocation at t=1,
          "E": objective value,
          "success": solver success flag,
          "message": solver message
        }
    """
    N = len(mu0)
    
    def objective(z):
        x0, x1 = z[:N], z[N:]
        E = (c0 * (x0 @ (Sigma0 @ x0) - lamb * (mu0 @ x0)) +
             c1 * (x1 @ (Sigma1 @ x1) - lamb * (mu1 @ x1)) +
             eta0 * np.sum((x1 - x0)**2))
        return E
    
    # Initial guess: uniform portfolio
    if x0_init is None:
        x0_init = np.ones(N) / N
    if x1_init is None:
        x1_init = np.ones(N) / N
    z0 = np.concatenate([x0_init, x1_init])
    
    # Constraints: sum(x0) = 1, sum(x1) = 1
    cons = [
        {"type": "eq", "fun": lambda z: np.sum(z[:N]) - 1.0},
        {"type": "eq", "fun": lambda z: np.sum(z[N:]) - 1.0}
    ]
    
    # Bounds: non-negativity
    bounds = [(0, None)] * (2 * N)
    
    res = minimize(objective, z0, method="SLSQP",
                   constraints=cons, bounds=bounds,
                   options={"ftol": 1e-9, "disp": False, "maxiter": 1000})
    
    x0, x1 = res.x[:N], res.x[N:]
    return {"x0": x0, "x1": x1, "E": res.fun,
            "success": res.success, "message": res.message}
    
