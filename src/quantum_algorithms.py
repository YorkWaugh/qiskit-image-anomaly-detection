from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator  # Use AerSimulator for simulation
import numpy as np
from scipy.stats import entropy as kl_divergence  # For KL Divergence
from tqdm import tqdm


class QuantumAnomalyDetection:
    """Quantum anomaly detection: learn profile and score via JSD."""

    def __init__(self, num_qubits, shots=1024, anomaly_threshold=0.5):
        self.num_qubits = num_qubits
        self.shots = shots
        self.simulator = AerSimulator()  # use AerSimulator
        self.normal_profile = None
        self.anomaly_threshold = anomaly_threshold

    def learn_normal_profile(self, list_of_encoded_normal_circuits):
        """Average counts from normal circuits to build profile."""
        if not list_of_encoded_normal_circuits:
            print(
                "Warning: No normal circuits provided to learn profile. Profile remains uninitialized."
            )
            self.normal_profile = {}
            return

        aggregated_probabilities = {}
        num_normal_samples = len(list_of_encoded_normal_circuits)

        print(f"Learning normal profile from {num_normal_samples} normal samples...")
        for i, encoded_qc in enumerate(
            tqdm(
                list_of_encoded_normal_circuits,
                desc="Processing Normal Profile Circuits",
            )
        ):
            qc_to_run = encoded_qc.copy()
            measurable_qc = self.create_feature_map_circuit(qc_to_run)
            counts = self.run_quantum_circuit(measurable_qc)

            for state, num_occurrences in counts.items():
                prob = num_occurrences / self.shots
                aggregated_probabilities[state] = (
                    aggregated_probabilities.get(state, 0.0) + prob
                )

        if num_normal_samples > 0:
            self.normal_profile = {
                state: total_prob / num_normal_samples
                for state, total_prob in aggregated_probabilities.items()
            }
        else:
            self.normal_profile = {}

        print("Normal profile learned.")

    def create_feature_map_circuit(self, data_vector):
        """Append RY/CNOT/RZ layers and measure the circuit."""
        if not isinstance(data_vector, QuantumCircuit):
            raise TypeError(
                "Input 'data_vector' must be a QuantumCircuit from ImageEncoder."
            )

        qc = data_vector
        num_qubits = qc.num_qubits

        if num_qubits > 0:
            fixed_rotation_angle_ry1 = np.pi / 2
            for qubit_index in range(num_qubits):
                qc.ry(fixed_rotation_angle_ry1, qubit_index)

        if num_qubits > 1:
            # linear CNOT layer
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)

        if num_qubits > 0:
            # RZ layer
            fixed_rotation_angle_rz = np.pi / 3
            for qubit_index in range(num_qubits):
                qc.rz(fixed_rotation_angle_rz, qubit_index)

        if num_qubits > 1:
            # reverse CNOT layer
            for i in range(num_qubits - 1, 0, -1):
                qc.cx(i, i - 1)

        if num_qubits > 0:
            # RY layer
            fixed_rotation_angle_ry2 = np.pi / 4
            for qubit_index in range(num_qubits):
                qc.ry(fixed_rotation_angle_ry2, qubit_index)

        if num_qubits > 0:
            # second RZ layer
            fixed_rotation_angle_rz2 = np.pi / 5
            for qubit_index in range(num_qubits):
                qc.rz(fixed_rotation_angle_rz2, qubit_index)

        if num_qubits > 1:
            # circular CNOT layer
            for i in range(num_qubits):
                qc.cx(i, (i + 1) % num_qubits)

        if num_qubits > 0:
            # final RY layer
            fixed_rotation_angle_ry3 = np.pi / 6
            for qubit_index in range(num_qubits):
                qc.ry(fixed_rotation_angle_ry3, qubit_index)

        qc.measure_all()
        return qc

    def run_quantum_circuit(self, quantum_circuit):
        """Run circuit on AerSimulator and return counts."""
        if not isinstance(quantum_circuit, QuantumCircuit):
            raise TypeError("Input 'quantum_circuit' must be a QuantumCircuit.")
        if self.simulator is None:
            raise RuntimeError("Simulator not initialized. Cannot run the circuit.")

        compiled_circuit = transpile(quantum_circuit, self.simulator)
        result = self.simulator.run(compiled_circuit, shots=self.shots).result()
        counts = result.get_counts(compiled_circuit)
        return counts

    def analyze_results(self, counts):
        """Compute normalized JSD between counts and normal profile."""
        if self.normal_profile is None:
            print(
                "Error: Normal profile has not been learned. Cannot analyze results. Returning max anomaly score (1.0)."
            )
            return 1.0
        if not self.normal_profile:
            print(
                "Warning: Normal profile is empty. Returning max anomaly score (1.0)."
            )
            return 1.0

        current_probabilities = {
            state: count / self.shots for state, count in counts.items()
        }

        all_states = set(self.normal_profile.keys()) | set(current_probabilities.keys())

        p = np.array([self.normal_profile.get(state, 0.0) for state in all_states])
        q = np.array([current_probabilities.get(state, 0.0) for state in all_states])

        p_sum = np.sum(p)
        q_sum = np.sum(q)
        if p_sum == 0 or q_sum == 0:
            return 1.0

        p_normalized = p / p_sum
        q_normalized = q / q_sum

        m = 0.5 * (p_normalized + q_normalized)
        jsd = 0.5 * (kl_divergence(p_normalized, m) + kl_divergence(q_normalized, m))

        if np.isinf(jsd):
            return 1.0
        if np.isnan(jsd):
            if np.allclose(p_normalized, q_normalized):
                return 0.0
            else:
                print(
                    f"Warning: JSD calculation resulted in NaN. p={p_normalized}, q={q_normalized}. Returning max anomaly score."
                )
                return 1.0

        # JSD max is log(2); normalize to [0,1]
        normalized_jsd = jsd / np.log(2)
        return min(max(normalized_jsd, 0.0), 1.0)

    def is_anomalous(self, anomaly_score):
        """Return True if score exceeds threshold."""
        return anomaly_score > self.anomaly_threshold
