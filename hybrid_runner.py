import os
import json
import numpy as np
import pandas as pd

from braket.circuits import Circuit
from braket.aws import AwsDevice
from braket.jobs.metrics import log_metric


# ---------------------------------------------------------
# Convert sparse QUBO → dense Ising
# ---------------------------------------------------------

def qubo_to_ising_sparse(df: pd.DataFrame):
    # indices assumed 0..n-1
    n = int(max(df["i"].max(), df["j"].max()) + 1)
    h = np.zeros(n)
    J = np.zeros((n, n))

    for _, row in df.iterrows():
        i = int(row["i"])
        j = int(row["j"])
        q = float(row["value"])

        if i == j:
            # linear term
            h[i] += q
        else:
            # symmetric coupling
            J[i, j] += q / 2.0
            J[j, i] += q / 2.0

    return h, J


# ---------------------------------------------------------
# QAOA circuit with fixed angles (numeric parameters only)
# ---------------------------------------------------------

def build_qaoa_fixed(p: int, h: np.ndarray, J: np.ndarray) -> Circuit:
    n = len(h)
    circ = Circuit().h(range(n))

    # hard-coded parameters so the hybrid job doesn't need free params
    gamma = 0.8
    beta = 0.6

    for _layer in range(p):
        # cost Hamiltonian
        for i in range(n):
            if h[i] != 0.0:
                circ.rz(i, 2.0 * gamma * h[i])

        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0.0:
                    circ.cnot(i, j)
                    circ.rz(j, 2.0 * gamma * J[i, j])
                    circ.cnot(i, j)

        # mixer
        for i in range(n):
            circ.rx(i, 2.0 * beta)

    # measure all qubits explicitly (no measure_all() in this Braket version)
    for i in range(n):
        circ.measure(i)

    return circ


# ---------------------------------------------------------
# Hybrid Job entry point
# ---------------------------------------------------------

from braket.jobs import save_job_result

def run(**kwargs):
    print("=== Amazon Braket Hybrid Job Started ===")

    p = int(kwargs.get("p", 1))
    shots = int(kwargs.get("shots", 500))
    print(f"Hyperparameters → p={p}, shots={shots}")

    # Where Braket mounts input
    input_dir = os.environ.get("AMZN_BRAKET_INPUT_DIR", "/opt/braket/input/data")
    input_path = os.path.join(input_dir, "input")
    print("Hybrid Job Input Directory:", input_path)

    qubo_path = os.path.join(input_path, "qubo_sparse.csv")
    print("Loading sparse QUBO:", qubo_path)

    if not os.path.exists(qubo_path):
        raise FileNotFoundError(f"QUBO file not found at {qubo_path}")

    df = pd.read_csv(qubo_path)

    print("Converting QUBO → Ising")
    h, J = qubo_to_ising_sparse(df)
    n = len(h)
    print(f"Problem size → {n} qubits")

    # Safety gate for TN1
    if n > 50:
        raise ValueError(f"Reduced QUBO still has {n} qubits; TN1 supports ~50. Reduce MAX_VARS.")

    print("Building QAOA Circuit...")
    circuit = build_qaoa_fixed(p, h, J)

    # Use Braket device
    device_arn = os.environ["AMZN_BRAKET_DEVICE_ARN"]
    print("Using device:", device_arn)
    device = AwsDevice(device_arn)

    print("Running circuit on device...")
    task = device.run(circuit, shots=shots)
    result = task.result()

    counts = {str(bitstring): int(count)
              for bitstring, count in result.measurement_counts.items()}

    print("Measurement counts:", counts)

    save_job_result({
        "counts": counts,
        "p": p,
        "shots": shots,
        "n_qubits": n
    })

    print("=== Hybrid Job DONE ===")
    return {"counts": counts, "p": p, "shots": shots, "n_qubits": n}

