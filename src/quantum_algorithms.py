from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
from scipy.stats import entropy as kl_divergence


class QuantumAnomalyDetection:
    def __init__(self, num_qubits, shots=1024):
        self.num_qubits = num_qubits
        self.shots = shots
        self.simulator = AerSimulator()
        self.normal_profile = None

    def learn_normal_profile(self, list_of_encoded_normal_circuits):
        if not list_of_encoded_normal_circuits:
            print(
                "Warning: No normal circuits provided to learn profile. Profile remains uninitialized."
            )
            self.normal_profile = {}
            return

        aggregated_probabilities = {}
        num_normal_samples = len(list_of_encoded_normal_circuits)

        print(f"Learning normal profile from {num_normal_samples} normal samples...")
        for i, encoded_qc in enumerate(list_of_encoded_normal_circuits):
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
        if not isinstance(data_vector, QuantumCircuit):
            raise TypeError(
                "Input 'data_vector' must be a QuantumCircuit from ImageEncoder."
            )

        qc = data_vector
        qc.measure_all()
        return qc

    def run_quantum_circuit(self, quantum_circuit):
        if not isinstance(quantum_circuit, QuantumCircuit):
            raise TypeError("Input 'quantum_circuit' must be a QuantumCircuit.")
        if self.simulator is None:
            raise RuntimeError("Simulator not initialized. Cannot run the circuit.")

        compiled_circuit = transpile(quantum_circuit, self.simulator)

        result = self.simulator.run(compiled_circuit, shots=self.shots).result()
        counts = result.get_counts(compiled_circuit)
        return counts

    def analyze_results(self, counts):
        if self.normal_profile is None:
            print(
                "Error: Normal profile has not been learned. Cannot analyze results. Returning max anomaly score (1.0)."
            )
            return 1.0
        if not self.normal_profile:
            print(
                "Warning: Normal profile is empty (e.g., no normal samples were processed for profile learning). \
                Treating current sample as maximally anomalous. Returning max anomaly score (1.0)."
            )
            return 1.0

        current_probabilities = {
            state: num_occurrences / self.shots
            for state, num_occurrences in counts.items()
        }

        if not current_probabilities and not self.normal_profile:
            return 0.0
        all_states = sorted(
            list(set(self.normal_profile.keys()) | set(current_probabilities.keys()))
        )

        if not all_states:
            return 0.0

        pk = np.array([current_probabilities.get(s, 0.0) for s in all_states])
        qk = np.array([self.normal_profile.get(s, 0.0) for s in all_states])

        mk = 0.5 * (pk + qk)

        jsd_part1 = kl_divergence(pk, mk)
        jsd_part2 = kl_divergence(qk, mk)

        if np.isinf(jsd_part1) or np.isinf(jsd_part2):
            return 1.0

        js_divergence = 0.5 * (jsd_part1 + jsd_part2)

        max_jsd = np.log(2)
        if max_jsd == 0:
            return 0.0

        normalized_jsd = js_divergence / max_jsd

        return np.clip(normalized_jsd, 0.0, 1.0)
