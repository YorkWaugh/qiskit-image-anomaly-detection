from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator  # Use AerSimulator for simulation
import numpy as np
from scipy.stats import entropy as kl_divergence  # For KL Divergence


class QuantumAnomalyDetection:
    """Handles quantum aspects of anomaly detection, including profile learning and scoring.

    Attributes:
        num_qubits (int): The number of qubits for the quantum circuits.
        shots (int): The number of times to run each quantum circuit for measurement.
        simulator (AerSimulator): The Qiskit Aer simulator instance.
        normal_profile (dict or None): A dictionary representing the probability distribution
                                       of measurement outcomes for normal data. Keys are bitstrings
                                       (e.g., '001'), and values are their probabilities.
                                       It's None until `learn_normal_profile` is successfully called.
    """

    def __init__(self, num_qubits, shots=1024):
        self.num_qubits = num_qubits
        self.shots = shots
        self.simulator = AerSimulator()  # Using Qiskit Aer's high-performance simulator
        self.normal_profile = None  # Will store the learned profile of normal data

    def learn_normal_profile(self, list_of_encoded_normal_circuits):
        """
        Learns a profile for 'normal' data by averaging the measurement outcome
        probabilities from a list of quantum circuits representing normal samples.

        Args:
            list_of_encoded_normal_circuits (list[QuantumCircuit]): A list of quantum circuits,
                each prepared with the state of a normal, encoded image.
        """
        if not list_of_encoded_normal_circuits:
            print(
                "Warning: No normal circuits provided to learn profile. Profile remains uninitialized."
            )
            self.normal_profile = (
                {}
            )  # Initialize to an empty profile to avoid None checks later if desired
            return

        aggregated_probabilities = (
            {}
        )  # To sum probabilities for each state across all normal samples
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
        """
        Appends measurement operations to a quantum circuit prepared with a quantum state.

        For this implementation, it's assumed that `data_vector` is a QuantumCircuit
        instance that has already been prepared with the desired quantum state (e.g., by ImageEncoder).
        This method then appends a measurement operation to all qubits.

        Args:
            data_vector (QuantumCircuit): The input quantum circuit with a prepared state.

        Returns:
            QuantumCircuit: The input circuit with measurement operations added.

        Raises:
            TypeError: If `data_vector` is not a QuantumCircuit.
        """
        if not isinstance(data_vector, QuantumCircuit):
            raise TypeError(
                "Input 'data_vector' must be a QuantumCircuit from ImageEncoder."
            )

        qc = data_vector  # The circuit from ImageEncoder already has the state prepared

        # Add measurement to all qubits. This implicitly adds classical bits.
        qc.measure_all()
        return qc

    def run_quantum_circuit(self, quantum_circuit):
        """
        Executes a given quantum circuit on the configured AerSimulator.

        Args:
            quantum_circuit (QuantumCircuit): The quantum circuit to run.

        Returns:
            dict: A dictionary of measurement counts (e.g., {'001': 512, '101': 512}).

        Raises:
            TypeError: If `quantum_circuit` is not a QuantumCircuit.
            RuntimeError: If the simulator has not been initialized.
        """
        if not isinstance(quantum_circuit, QuantumCircuit):
            raise TypeError("Input 'quantum_circuit' must be a QuantumCircuit.")
        if self.simulator is None:
            # This case should ideally not be reached if __init__ is always called.
            raise RuntimeError("Simulator not initialized. Cannot run the circuit.")

        # Transpile the circuit for optimal execution on the simulator
        compiled_circuit = transpile(quantum_circuit, self.simulator)

        # Run the simulation
        result = self.simulator.run(compiled_circuit, shots=self.shots).result()
        counts = result.get_counts(compiled_circuit)
        return counts

    def analyze_results(self, counts):
        """
        Analyzes the measurement counts of a quantum circuit against the learned `normal_profile`
        using Jensen-Shannon Divergence (JSD).

        The JSD provides a symmetric measure of similarity between two probability distributions.
        A higher JSD indicates greater dissimilarity (more anomalous).
        The raw JSD is normalized to the range [0, 1], where 1 represents maximal dissimilarity.

        Args:
            counts (dict): Measurement counts from a quantum circuit (e.g., {'001': 500, '110': 524}).

        Returns:
            float: A normalized anomaly score between 0 (matches normal profile) and 1 (maximally different).
        """
        if self.normal_profile is None:
            print(
                "Error: Normal profile has not been learned. Cannot analyze results. Returning max anomaly score (1.0)."
            )
            return 1.0
        if (
            not self.normal_profile
        ):  # Check if normal_profile is empty (e.g. no normal samples provided)
            print(
                "Warning: Normal profile is empty (e.g., no normal samples were processed for profile learning). \
                Treating current sample as maximally anomalous. Returning max anomaly score (1.0)."
            )
            return 1.0

        current_probabilities = {
            state: num_occurrences / self.shots
            for state, num_occurrences in counts.items()
        }

        # Case 1: Both profiles are empty (e.g. current_probabilities from a zero-shot run and empty normal_profile)
        # This case should ideally be caught by the `not self.normal_profile` check above if normal_profile is truly empty.
        # If current_probabilities is also empty, and normal_profile was not, it implies divergence.
        if not current_probabilities and not self.normal_profile:
            return 0.0  # Or handle as a specific case, e.g. if normal_profile was also empty due to no training data

        # Determine all unique states observed in either the current sample or the normal profile
        all_states = sorted(
            list(set(self.normal_profile.keys()) | set(current_probabilities.keys()))
        )

        if not all_states:  # Should be covered by the above, but as a safeguard
            return 0.0

        # Create probability vectors aligned by all_states, ensuring all states are represented.
        pk = np.array(
            [current_probabilities.get(s, 0.0) for s in all_states]
        )  # Current sample
        qk = np.array(
            [self.normal_profile.get(s, 0.0) for s in all_states]
        )  # Normal profile

        # Normalize pk and qk to ensure they are valid probability distributions over `all_states`.
        # This step is crucial if pk or qk were derived from counts that didn't sum to `self.shots`
        # or if only a subset of states was present in one of the distributions.
        # However, `current_probabilities` and `self.normal_profile` should already be normalized by `self.shots`
        # or by averaging over samples. If they are already sum-to-1 distributions over their respective observed states,
        # this re-normalization over `all_states` might not be strictly necessary but ensures robustness.
        # For JSD, the input vectors to kl_divergence must be probability distributions.

        # Calculate the mixed distribution M = 0.5 * (P + Q)
        mk = 0.5 * (pk + qk)

        # Calculate KL divergences: KLD(P || M) and KLD(Q || M)
        # scipy.stats.entropy calculates KL divergence if qk is provided (entropy(pk, qk))
        jsd_part1 = kl_divergence(pk, mk)  # KLD(P || M)
        jsd_part2 = kl_divergence(qk, mk)  # KLD(Q || M)

        # Handle potential -inf or inf results from kl_divergence if mk contains zeros where pk or qk do not.
        # kl_divergence handles pk[i] == 0 correctly (term is 0).
        # If mk[i] == 0 and pk[i] != 0 (or qk[i] != 0), it results in inf.
        if np.isinf(jsd_part1) or np.isinf(jsd_part2):
            # This indicates maximal divergence because M has zero probability where P or Q does not.
            return 1.0

        js_divergence = 0.5 * (jsd_part1 + jsd_part2)

        # Normalize JSD to [0, 1]. The maximum value of JSD (using natural log) is np.log(2).
        max_jsd = np.log(2)
        if max_jsd == 0:  # Should not happen with np.log(2)
            return 0.0  # Or raise an error, as this implies log(2) is zero.

        normalized_jsd = js_divergence / max_jsd

        # Clamp the result to [0, 1] to handle potential floating-point inaccuracies.
        return np.clip(normalized_jsd, 0.0, 1.0)
