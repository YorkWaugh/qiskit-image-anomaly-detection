import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

import config


def create_frqi_circuit(normalized_image_flat_vector):
    if len(normalized_image_flat_vector) != (config.IMG_SIZE * config.IMG_SIZE):
        raise ValueError(
            f"Image vector length {len(normalized_image_flat_vector)} does not match expected {config.IMG_SIZE*config.IMG_SIZE}"
        )

    qc = QuantumCircuit(config.TOTAL_QUBITS)
    position_qubits = list(range(config.NUM_POSITION_QUBITS))
    intensity_qubit = config.INTENSITY_QUBIT_INDEX

    ancilla_qubits_list = []
    if config.NUM_ANCILLA_QUBITS > 0:
        ancilla_start_index = config.NUM_POSITION_QUBITS + 1
        ancilla_qubits_indices = list(
            range(ancilla_start_index, ancilla_start_index + config.NUM_ANCILLA_QUBITS)
        )
        ancilla_qubits_list = [qc.qubits[i] for i in ancilla_qubits_indices]

    for i in position_qubits:
        qc.h(i)

    for pixel_idx, pixel_val_norm in enumerate(normalized_image_flat_vector):
        if pixel_val_norm > 1e-6:
            theta_for_mcry = np.pi * pixel_val_norm
            binary_pixel_idx = format(pixel_idx, f"0{config.NUM_POSITION_QUBITS}b")

            for i, bit in enumerate(binary_pixel_idx):
                if bit == "0":
                    qc.x(position_qubits[config.NUM_POSITION_QUBITS - 1 - i])

            control_q_objects = [qc.qubits[j] for j in position_qubits]
            q_ancillae_for_mcry = (
                ancilla_qubits_list if config.NUM_ANCILLA_QUBITS > 0 else None
            )

            qc.mcry(
                theta_for_mcry,
                control_q_objects,
                qc.qubits[intensity_qubit],
                q_ancillae=q_ancillae_for_mcry,
                mode="basic",
            )

            for i, bit in enumerate(binary_pixel_idx):
                if bit == "0":
                    qc.x(position_qubits[config.NUM_POSITION_QUBITS - 1 - i])
    return qc


def add_feature_extraction_ansatz(qc):
    for i in range(config.TOTAL_QUBITS):
        qc.ry(np.pi / 4, i)
    for i in range(config.TOTAL_QUBITS - 1):
        qc.cx(i, i + 1)
    if config.TOTAL_QUBITS > 1:
        qc.cx(config.TOTAL_QUBITS - 1, 0)
    return qc


def get_quantum_features(qc, simulator):
    observables = []
    for i in range(config.TOTAL_QUBITS):
        obs_str_list = ["I"] * config.TOTAL_QUBITS
        obs_str_list[config.TOTAL_QUBITS - 1 - i] = "Z"
        obs_str = "".join(obs_str_list)
        observables.append(SparsePauliOp(obs_str))

    qc_copy = qc.copy()
    qc_copy.save_statevector()

    transpiled_qc = transpile(qc_copy, simulator)
    result = simulator.run(transpiled_qc).result()
    statevector = result.get_statevector(transpiled_qc)

    exp_values = [statevector.expectation_value(obs).real for obs in observables]
    return np.array(exp_values)
