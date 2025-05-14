from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator  # Use AerSimulator for simulation
import numpy as np
from scipy.stats import entropy as kl_divergence  # For KL Divergence
from tqdm import tqdm


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
        anomaly_threshold (float): The threshold used to classify a score as anomalous.
    """

    def __init__(self, num_qubits, shots=1024, anomaly_threshold=0.5):
        self.num_qubits = num_qubits
        self.shots = shots
        self.simulator = AerSimulator()  # Using Qiskit Aer's high-performance simulator
        self.normal_profile = None  # Will store the learned profile of normal data
        self.anomaly_threshold = anomaly_threshold  # Store the threshold

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
        """
        Appends an enhanced feature map layer and measurement operations to a quantum circuit
        that has been prepared with an initial quantum state (e.g., by ImageEncoder).

        The feature map layer consists of single-qubit RY rotations followed by
        linear CNOT entanglement.

        Args:
            data_vector (QuantumCircuit): The input quantum circuit with a prepared state.

        Returns:
            QuantumCircuit: The circuit with the added feature map layer and measurement operations.

        Raises:
            TypeError: If `data_vector` is not a QuantumCircuit.
        """
        if not isinstance(data_vector, QuantumCircuit):
            raise TypeError(
                "Input 'data_vector' must be a QuantumCircuit from ImageEncoder."
            )

        qc = data_vector  # Use the circuit passed from ImageEncoder (or its copy)

        num_qubits = qc.num_qubits

        if num_qubits > 0:  # Ensure there are qubits in the circuit
            # Layer 1: RY rotations
            fixed_rotation_angle_ry1 = np.pi / 2
            for qubit_index in range(num_qubits):
                qc.ry(fixed_rotation_angle_ry1, qubit_index)

        if num_qubits > 1:
            # Layer 2: CNOT entanglement (linear)
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)

        if num_qubits > 0:
            # Layer 3: RZ rotations
            fixed_rotation_angle_rz = np.pi / 3  # Example angle, can be tuned
            for qubit_index in range(num_qubits):
                qc.rz(fixed_rotation_angle_rz, qubit_index)

        if num_qubits > 1:
            # Layer 4: CNOT entanglement (reverse linear or circular if preferred)
            # Example: reverse linear
            for i in range(num_qubits - 1, 0, -1):
                qc.cx(i, i - 1)
            # Alternative: another linear layer, or a different pattern
            # for i in range(num_qubits - 1):
            #     qc.cx(i, (i + 1) % num_qubits) # Example of a circular CNOT

        if num_qubits > 0:
            # Layer 5: RY rotations
            fixed_rotation_angle_ry2 = np.pi / 4  # Example angle, can be tuned
            for qubit_index in range(num_qubits):
                qc.ry(fixed_rotation_angle_ry2, qubit_index)

        # Add more layers for increased complexity
        if num_qubits > 0:
            # Layer 6: RZ rotations
            fixed_rotation_angle_rz2 = np.pi / 5  # Example angle
            for qubit_index in range(num_qubits):
                qc.rz(fixed_rotation_angle_rz2, qubit_index)

        if num_qubits > 1:
            # Layer 7: CNOT entanglement (circular)
            for i in range(num_qubits):
                qc.cx(i, (i + 1) % num_qubits)

        if num_qubits > 0:
            # Layer 8: RY rotations
            fixed_rotation_angle_ry3 = np.pi / 6  # Example angle
            for qubit_index in range(num_qubits):
                qc.ry(fixed_rotation_angle_ry3, qubit_index)

        # Measurement
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
                "Warning: Normal profile is empty (e.g., no normal samples were processed for profile learning). "
                "Returning max anomaly score (1.0)."
            )
            return 1.0

        # Convert current counts to a probability distribution
        current_probabilities = {
            state: count / self.shots for state, count in counts.items()
        }

        # Ensure both distributions cover the same states, padding with zero probability where necessary
        all_states = set(self.normal_profile.keys()) | set(current_probabilities.keys())

        p = np.array([self.normal_profile.get(state, 0.0) for state in all_states])
        q = np.array([current_probabilities.get(state, 0.0) for state in all_states])

        # Normalize p and q to ensure they sum to 1 (important for KL divergence)
        # This handles cases where some states might not have been observed due to shot noise,
        # or if the normal_profile wasn't perfectly normalized initially.
        p_sum = np.sum(p)
        q_sum = np.sum(q)
        if p_sum == 0 or q_sum == 0:
            # This can happen if one distribution is all zeros (e.g., empty normal profile or no counts)
            # If normal_profile is empty, we already returned 1.0.
            # If current_probabilities is empty (no counts), it implies maximal difference from any non-empty profile.
            return 1.0  # Max anomaly if one distribution is empty and the other is not (or both empty)

        p_normalized = p / p_sum
        q_normalized = q / q_sum

        # Calculate Jensen-Shannon Divergence (JSD)
        m = 0.5 * (p_normalized + q_normalized)
        jsd = 0.5 * (kl_divergence(p_normalized, m) + kl_divergence(q_normalized, m))

        if np.isinf(jsd):
            # This can happen if, for some state i, m[i] is 0 but p_normalized[i] or q_normalized[i] is not.
            # This implies a state exists in one distribution but not in the average, which is unlikely if m is calculated correctly.
            # More likely, if a state is in P but not Q (or vice versa), then one of the KL terms might be large.
            # If JSD is infinite, it implies maximal divergence.
            return 1.0
        if np.isnan(jsd):
            # Can happen if p and q are identical, kl_divergence(p,p) can be 0 or nan depending on exact zero handling.
            # If they are identical, JSD should be 0.
            # A more robust check: if np.allclose(p_normalized, q_normalized), jsd = 0
            if np.allclose(p_normalized, q_normalized):
                return 0.0
            else:
                # If it's NaN for other reasons, it's an issue, treat as max anomaly for safety.
                print(
                    f"Warning: JSD calculation resulted in NaN. p={p_normalized}, q={q_normalized}. Returning max anomaly score."
                )
                return 1.0

        # Normalize JSD to be between 0 and 1. Max JSD is log(2).
        normalized_jsd = jsd / np.log(2)
        return min(max(normalized_jsd, 0.0), 1.0)  # Clamp to [0,1] to be safe

    def is_anomalous(self, anomaly_score):
        """Determines if a score indicates an anomaly based on the threshold."""
        return anomaly_score > self.anomaly_threshold
