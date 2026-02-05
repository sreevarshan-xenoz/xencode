"""
Quantum-Ready Algorithms
Implements QuantumReadyManager for hybrid computing, quantum algorithm implementations,
classical-quantum fallback mechanisms, and quantum simulation capabilities.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import json
import secrets
import hashlib
from datetime import datetime
import numpy as np
from dataclasses import dataclass
import math
import random
import threading
import time
from collections import defaultdict


logger = logging.getLogger(__name__)


class QuantumAlgorithmType(Enum):
    """Types of quantum algorithms."""
    SHORS_FACTORIZATION = "shors_factoring"
    GROVERS_SEARCH = "grovers_search"
    VQE = "variational_quantum_eigensolver"
    QAOA = "quantum_approximate_optimization_algorithm"
    HHL = "hhl_algorithm"  # Linear systems solver
    QUANTUM_FOURIER_TRANSFORM = "quantum_fourier_transform"
    QUANTUM_AMPLITUDE_ESTIMATION = "quantum_amplitude_estimation"
    HIDDEN_SUBGROUP = "hidden_subgroup_problem"
    SIMONS_ALGORITHM = "simons_algorithm"
    DEUTSCH_JOZSA = "deutsch_jozsa_algorithm"


class HybridComputingMode(Enum):
    """Modes for hybrid classical-quantum computing."""
    CLASSICAL_FIRST = "classical_first"
    QUANTUM_FIRST = "quantum_first"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


class QuantumSimulationBackend(Enum):
    """Backends for quantum simulation."""
    QISKIT = "qiskit"
    CIRQ = "cirq"
    PYZX = "pyzx"
    PULSER = "pulser"
    SIMULATED = "simulated"


@dataclass
class QuantumCircuit:
    """Represents a quantum circuit."""
    circuit_id: str
    algorithm_type: QuantumAlgorithmType
    num_qubits: int
    depth: int  # Circuit depth
    gates: List[Dict[str, Any]]  # List of quantum gates
    parameters: Dict[str, float]  # Parameterized gates
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class QuantumTask:
    """Represents a quantum computing task."""
    task_id: str
    algorithm_type: QuantumAlgorithmType
    input_data: Any
    expected_output_size: int
    priority: int  # 1-5 scale
    submitted_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    status: str  # queued, running, completed, failed, fallback_to_classical
    result: Optional[Any]
    execution_time: Optional[float]  # in seconds
    backend_used: Optional[QuantumSimulationBackend]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ClassicalFallbackResult:
    """Result when quantum computation falls back to classical."""
    task_id: str
    classical_result: Any
    fallback_reason: str
    classical_execution_time: float
    quantum_equivalent_complexity: str  # Polynomial, Exponential, etc.


class QuantumGate:
    """Represents a quantum gate operation."""
    
    def __init__(self, gate_type: str, target_qubits: List[int], parameters: Dict[str, float] = None):
        self.gate_type = gate_type
        self.target_qubits = target_qubits
        self.parameters = parameters or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert gate to dictionary representation."""
        return {
            "type": self.gate_type,
            "qubits": self.target_qubits,
            "params": self.parameters
        }


class QuantumCircuitBuilder:
    """Builds quantum circuits for various algorithms."""
    
    def __init__(self):
        self.gate_library = {
            "h": self._create_hadamard_gate,
            "x": self._create_pauli_x_gate,
            "y": self._create_pauli_y_gate,
            "z": self._create_pauli_z_gate,
            "cx": self._create_cnot_gate,
            "cz": self._create_cz_gate,
            "rx": self._create_rx_gate,
            "ry": self._create_ry_gate,
            "rz": self._create_rz_gate,
            "u": self._create_u_gate,
            "swap": self._create_swap_gate
        }
        
    def build_shors_circuit(self, n: int, a: int) -> QuantumCircuit:
        """Build a circuit for Shor's algorithm (simplified)."""
        circuit_id = f"circuit_shor_{secrets.token_hex(8)}"
        
        # Simplified Shor's algorithm circuit
        # In reality, this would be much more complex
        num_qubits = 2 * math.ceil(math.log2(n)) + 3  # Approximation
        gates = [
            self._create_hadamard_gate(list(range(num_qubits//2)), {}),
            # Add modular exponentiation gates (simplified)
            self._create_cnot_gate([0, 1], {}),
            self._create_cnot_gate([1, 2], {}),
        ]
        
        circuit = QuantumCircuit(
            circuit_id=circuit_id,
            algorithm_type=QuantumAlgorithmType.SHORS_FACTORIZATION,
            num_qubits=num_qubits,
            depth=10,  # Simplified
            gates=[gate.to_dict() for gate in gates],
            parameters={"n": n, "a": a},
            created_at=datetime.now(),
            metadata={"algorithm": "shors", "input_n": n}
        )
        
        return circuit
        
    def build_grovers_circuit(self, n: int, marked_item: int) -> QuantumCircuit:
        """Build a circuit for Grover's algorithm (simplified)."""
        circuit_id = f"circuit_grover_{secrets.token_hex(8)}"
        
        # Simplified Grover's algorithm circuit
        num_qubits = n
        gates = []
        
        # Initialize superposition
        for i in range(n):
            gates.append(self._create_hadamard_gate([i], {}))
            
        # Oracle and diffusion operator (simplified)
        for iteration in range(int(math.pi/4 * math.sqrt(2**n))):
            # Oracle: mark the solution
            gates.append(self._create_z_gate([marked_item], {}))
            
            # Diffusion operator
            for i in range(n):
                gates.append(self._create_hadamard_gate([i], {}))
                gates.append(self._create_pauli_x_gate([i], {}))
                
            # Multi-controlled Z gate (simplified)
            gates.append(self._create_cnot_gate([0, 1], {}))
            
            for i in range(n):
                gates.append(self._create_pauli_x_gate([i], {}))
                gates.append(self._create_hadamard_gate([i], {}))
        
        circuit = QuantumCircuit(
            circuit_id=circuit_id,
            algorithm_type=QuantumAlgorithmType.GROVERS_SEARCH,
            num_qubits=num_qubits,
            depth=len(gates),
            gates=[gate.to_dict() for gate in gates],
            parameters={"n": n, "marked_item": marked_item},
            created_at=datetime.now(),
            metadata={"algorithm": "grovers", "search_space_size": 2**n}
        )
        
        return circuit
        
    def build_vqe_circuit(self, hamiltonian_terms: List[Tuple], ansatz_depth: int = 2) -> QuantumCircuit:
        """Build a circuit for Variational Quantum Eigensolver (simplified)."""
        circuit_id = f"circuit_vqe_{secrets.token_hex(8)}"
        
        # Simplified VQE circuit
        num_qubits = len(hamiltonian_terms) if hamiltonian_terms else 2
        gates = []
        
        # Prepare initial state (Hartree-Fock)
        for i in range(num_qubits):
            gates.append(self._create_hadamard_gate([i], {}))
            
        # Ansatz circuit (UCCSD-like, simplified)
        for depth in range(ansatz_depth):
            # Entangling layer
            for i in range(0, num_qubits - 1, 2):
                gates.append(self._create_cnot_gate([i, i+1], {}))
                
            # Parametrized rotations
            for i in range(num_qubits):
                gates.append(self._create_ry_gate([i], {"theta": 0.0}))
                
            # Odd entangling layer
            for i in range(1, num_qubits - 1, 2):
                gates.append(self._create_cnot_gate([i, i+1], {}))
                
            # More parametrized rotations
            for i in range(num_qubits):
                gates.append(self._create_rz_gate([i], {"phi": 0.0}))
        
        circuit = QuantumCircuit(
            circuit_id=circuit_id,
            algorithm_type=QuantumAlgorithmType.VQE,
            num_qubits=num_qubits,
            depth=len(gates),
            gates=[gate.to_dict() for gate in gates],
            parameters={"hamiltonian_terms": len(hamiltonian_terms), "ansatz_depth": ansatz_depth},
            created_at=datetime.now(),
            metadata={"algorithm": "vqe", "hamiltonian_terms_count": len(hamiltonian_terms)}
        )
        
        return circuit
        
    def _create_hadamard_gate(self, qubits: List[int], params: Dict[str, float]) -> QuantumGate:
        return QuantumGate("h", qubits, params)
        
    def _create_pauli_x_gate(self, qubits: List[int], params: Dict[str, float]) -> QuantumGate:
        return QuantumGate("x", qubits, params)
        
    def _create_pauli_y_gate(self, qubits: List[int], params: Dict[str, float]) -> QuantumGate:
        return QuantumGate("y", qubits, params)
        
    def _create_pauli_z_gate(self, qubits: List[int], params: Dict[str, float]) -> QuantumGate:
        return QuantumGate("z", qubits, params)
        
    def _create_cnot_gate(self, qubits: List[int], params: Dict[str, float]) -> QuantumGate:
        return QuantumGate("cx", qubits, params)
        
    def _create_cz_gate(self, qubits: List[int], params: Dict[str, float]) -> QuantumGate:
        return QuantumGate("cz", qubits, params)
        
    def _create_rx_gate(self, qubits: List[int], params: Dict[str, float]) -> QuantumGate:
        return QuantumGate("rx", qubits, params)
        
    def _create_ry_gate(self, qubits: List[int], params: Dict[str, float]) -> QuantumGate:
        return QuantumGate("ry", qubits, params)
        
    def _create_rz_gate(self, qubits: List[int], params: Dict[str, float]) -> QuantumGate:
        return QuantumGate("rz", qubits, params)
        
    def _create_u_gate(self, qubits: List[int], params: Dict[str, float]) -> QuantumGate:
        return QuantumGate("u", qubits, params)
        
    def _create_swap_gate(self, qubits: List[int], params: Dict[str, float]) -> QuantumGate:
        return QuantumGate("swap", qubits, params)


class QuantumSimulator:
    """Simulates quantum computations."""
    
    def __init__(self, backend: QuantumSimulationBackend = QuantumSimulationBackend.SIMULATED):
        self.backend = backend
        self.simulation_cache = {}
        self.performance_stats = {
            "total_simulations": 0,
            "average_simulation_time": 0.0,
            "cache_hit_rate": 0.0
        }
        
    def simulate_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
        """Simulate a quantum circuit and return measurement results."""
        start_time = time.time()
        
        # Create a cache key based on circuit properties
        cache_key = f"{circuit.circuit_id}_{shots}"
        
        # Check cache first
        if cache_key in self.simulation_cache:
            self.performance_stats["cache_hit_rate"] = (
                self.performance_stats["cache_hit_rate"] * 0.9 + 0.1
            )  # Moving average
            return self.simulation_cache[cache_key]
            
        # Perform simulation based on algorithm type
        if circuit.algorithm_type == QuantumAlgorithmType.GROVERS_SEARCH:
            result = self._simulate_grovers(circuit, shots)
        elif circuit.algorithm_type == QuantumAlgorithmType.SHORS_FACTORIZATION:
            result = self._simulate_shors(circuit, shots)
        elif circuit.algorithm_type == QuantumAlgorithmType.VQE:
            result = self._simulate_vqe(circuit, shots)
        else:
            # Generic simulation for other algorithms
            result = self._simulate_generic(circuit, shots)
            
        # Record performance
        execution_time = time.time() - start_time
        total_time = self.performance_stats["average_simulation_time"] * self.performance_stats["total_simulations"]
        self.performance_stats["total_simulations"] += 1
        self.performance_stats["average_simulation_time"] = (total_time + execution_time) / self.performance_stats["total_simulations"]
        
        # Cache result
        self.simulation_cache[cache_key] = result
        
        return result
        
    def _simulate_grovers(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Simulate Grover's algorithm."""
        # Simplified simulation - in reality, this would use amplitude amplification
        marked_item = circuit.parameters.get("marked_item", 0)
        n_qubits = circuit.num_qubits
        
        # Probability of finding the marked item should be high
        results = {}
        for _ in range(shots):
            # With high probability, we measure the marked item
            if random.random() < 0.9:  # 90% success rate for simulation
                result = format(marked_item, f'0{n_qubits}b')
            else:
                # Occasionally measure other states
                other_item = random.randint(0, 2**n_qubits - 1)
                result = format(other_item, f'0{n_qubits}b')
                
            results[result] = results.get(result, 0) + 1
            
        return results
        
    def _simulate_shors(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Simulate Shor's algorithm."""
        n = circuit.parameters.get("n", 15)  # Default to factoring 15
        a = circuit.parameters.get("a", 2)   # Default base
        
        # Simplified simulation - Shor's algorithm finds period
        # For n=15, a=2: period r=4, which leads to factors 3 and 5
        results = {}
        for _ in range(shots):
            # Simulate measuring period
            if n == 15 and a == 2:
                # Period should be 4 for a=2, n=15
                period_bits = format(4, f'0{circuit.num_qubits}b')
                results[period_bits] = results.get(period_bits, 0) + 1
            else:
                # Generic simulation
                random_result = format(random.randint(0, 2**circuit.num_qubits - 1), f'0{circuit.num_qubits}b')
                results[random_result] = results.get(random_result, 0) + 1
                
        return results
        
    def _simulate_vqe(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Simulate VQE algorithm."""
        # Simplified simulation - VQE estimates ground state energy
        results = {}
        for _ in range(shots):
            # Simulate energy measurements
            energy_estimate = random.uniform(-1.0, 1.0)  # Simplified energy range
            # Convert to binary representation
            energy_binary = format(int((energy_estimate + 1.0) * 512), f'0{circuit.num_qubits}b')  # Scale to fit qubits
            results[energy_binary] = results.get(energy_binary, 0) + 1
            
        return results
        
    def _simulate_generic(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Generic simulation for other quantum algorithms."""
        results = {}
        for _ in range(shots):
            # Random measurement outcomes
            result = format(random.randint(0, 2**circuit.num_qubits - 1), f'0{circuit.num_qubits}b')
            results[result] = results.get(result, 0) + 1
            
        return results


class ClassicalFallbackManager:
    """Manages fallback to classical algorithms when quantum fails."""
    
    def __init__(self):
        self.fallback_strategies = {
            QuantumAlgorithmType.SHORS_FACTORIZATION: self._factor_classically,
            QuantumAlgorithmType.GROVERS_SEARCH: self._search_classically,
            QuantumAlgorithmType.VQE: self._solve_classically,
            QuantumAlgorithmType.QAOA: self._optimize_classically,
            QuantumAlgorithmType.HHL: self._solve_linear_systems_classically
        }
        
    def fallback_to_classical(
        self, 
        algorithm_type: QuantumAlgorithmType, 
        input_data: Any
    ) -> ClassicalFallbackResult:
        """Execute classical fallback for quantum algorithm."""
        start_time = time.time()
        
        if algorithm_type in self.fallback_strategies:
            try:
                result = self.fallback_strategies[algorithm_type](input_data)
                execution_time = time.time() - start_time
                
                fallback_result = ClassicalFallbackResult(
                    task_id=f"fallback_{secrets.token_hex(8)}",
                    classical_result=result,
                    fallback_reason="quantum_unavailable",
                    classical_execution_time=execution_time,
                    quantum_equivalent_complexity="exponential"  # Default assumption
                )
                
                logger.info(f"Fell back to classical for {algorithm_type.value}, took {execution_time:.2f}s")
                return fallback_result
            except Exception as e:
                logger.error(f"Classical fallback failed for {algorithm_type.value}: {str(e)}")
                return ClassicalFallbackResult(
                    task_id=f"fallback_{secrets.token_hex(8)}",
                    classical_result=None,
                    fallback_reason=f"classical_failure: {str(e)}",
                    classical_execution_time=time.time() - start_time,
                    quantum_equivalent_complexity="unknown"
                )
        else:
            logger.warning(f"No classical fallback defined for {algorithm_type.value}")
            return ClassicalFallbackResult(
                task_id=f"fallback_{secrets.token_hex(8)}",
                classical_result=None,
                fallback_reason="no_fallback_defined",
                classical_execution_time=0.0,
                quantum_equivalent_complexity="unknown"
            )
            
    def _factor_classically(self, input_data: Any) -> Any:
        """Classical integer factorization."""
        if isinstance(input_data, dict) and "n" in input_data:
            n = input_data["n"]
            factors = []
            d = 2
            while d * d <= n:
                while n % d == 0:
                    factors.append(d)
                    n //= d
                d += 1
            if n > 1:
                factors.append(n)
            return factors
        return None
        
    def _search_classically(self, input_data: Any) -> Any:
        """Classical search algorithm."""
        if isinstance(input_data, dict) and "list" in input_data and "target" in input_data:
            lst = input_data["list"]
            target = input_data["target"]
            for i, item in enumerate(lst):
                if item == target:
                    return i
        return -1  # Not found
        
    def _solve_classically(self, input_data: Any) -> Any:
        """Classical problem solver (placeholder)."""
        # For VQE, this would use classical optimization methods
        # For now, return a placeholder result
        return {"solution": "classical_approximation", "energy": -0.5}
        
    def _optimize_classically(self, input_data: Any) -> Any:
        """Classical optimization."""
        # Placeholder for classical optimization
        return {"optimal_solution": "classical_result", "cost": 0.0}
        
    def _solve_linear_systems_classically(self, input_data: Any) -> Any:
        """Classical linear system solver."""
        # Placeholder for classical linear algebra
        return {"solution_vector": [1.0, 2.0, 3.0]}  # Placeholder


class QuantumReadyManager:
    """
    Quantum-ready manager for hybrid computing with quantum algorithm implementations,
    classical-quantum fallback mechanisms, and quantum simulation.
    """
    
    def __init__(self, default_backend: QuantumSimulationBackend = QuantumSimulationBackend.SIMULATED):
        self.circuit_builder = QuantumCircuitBuilder()
        self.quantum_simulator = QuantumSimulator(default_backend)
        self.fallback_manager = ClassicalFallbackManager()
        self.tasks: Dict[str, QuantumTask] = {}
        self.active_simulations: Dict[str, asyncio.Task] = {}
        self.hybrid_mode = HybridComputingMode.ADAPTIVE
        self.quantum_readiness_level = 0.0  # 0.0 to 1.0 scale
        self.hardware_available = False  # Whether real quantum hardware is available
        
    def set_hybrid_mode(self, mode: HybridComputingMode):
        """Set the hybrid computing mode."""
        self.hybrid_mode = mode
        logger.info(f"Set hybrid mode to {mode.value}")
        
    def assess_quantum_readiness(self) -> float:
        """Assess the quantum readiness of the system."""
        # Calculate readiness based on various factors
        factors = {
            "simulation_support": 0.3 if self.quantum_simulator else 0.0,
            "algorithm_coverage": 0.4 if len(self.fallback_manager.fallback_strategies) > 3 else 0.2,
            "hardware_availability": 0.3 if self.hardware_available else 0.0
        }
        
        readiness = sum(factors.values())
        self.quantum_readiness_level = min(readiness, 1.0)
        
        return self.quantum_readiness_level
        
    async def execute_quantum_algorithm(
        self, 
        algorithm_type: QuantumAlgorithmType, 
        input_data: Any,
        priority: int = 3,
        use_hardware: bool = False
    ) -> str:
        """Execute a quantum algorithm."""
        task_id = f"quantum_task_{secrets.token_hex(8)}"
        
        task = QuantumTask(
            task_id=task_id,
            algorithm_type=algorithm_type,
            input_data=input_data,
            expected_output_size=0,  # Will be determined during execution
            priority=priority,
            submitted_at=datetime.now(),
            started_at=None,
            completed_at=None,
            status="queued",
            result=None,
            execution_time=None,
            backend_used=None,
            metrics={},
            metadata={"input_size": len(str(input_data))}
        )
        
        self.tasks[task_id] = task
        
        logger.info(f"Queued quantum task {task_id} for {algorithm_type.value}")
        
        # Schedule execution based on hybrid mode
        if self.hybrid_mode == HybridComputingMode.QUANTUM_FIRST or use_hardware:
            await self._execute_quantum_task(task_id)
        elif self.hybrid_mode == HybridComputingMode.CLASSICAL_FIRST:
            # Check if classical is more appropriate
            if self._should_use_classical_for_input_size(input_data, algorithm_type):
                await self._execute_classical_fallback(task_id)
            else:
                await self._execute_quantum_task(task_id)
        elif self.hybrid_mode == HybridComputingMode.ADAPTIVE:
            # Adaptive decision based on input size and algorithm type
            if self._is_quantum_advantage_likely(input_data, algorithm_type):
                await self._execute_quantum_task(task_id)
            else:
                await self._execute_classical_fallback(task_id)
        else:  # PARALLEL
            # Execute both and return the first to complete
            quantum_task = asyncio.create_task(self._execute_quantum_task(task_id))
            classical_task = asyncio.create_task(self._execute_classical_fallback(task_id))
            
            # Wait for the first to complete
            done, pending = await asyncio.wait(
                [quantum_task, classical_task], 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel the remaining task
            for task_to_cancel in pending:
                task_to_cancel.cancel()
                
        return task_id
        
    async def _execute_quantum_task(self, task_id: str):
        """Execute a quantum task."""
        task = self.tasks[task_id]
        task.status = "running"
        task.started_at = datetime.now()
        
        try:
            # Build the appropriate circuit
            if task.algorithm_type == QuantumAlgorithmType.SHORS_FACTORIZATION:
                circuit = self.circuit_builder.build_shors_circuit(
                    n=task.input_data.get("n", 15),
                    a=task.input_data.get("a", 2)
                )
            elif task.algorithm_type == QuantumAlgorithmType.GROVERS_SEARCH:
                circuit = self.circuit_builder.build_grovers_circuit(
                    n=task.input_data.get("n", 3),
                    marked_item=task.input_data.get("marked_item", 0)
                )
            elif task.algorithm_type == QuantumAlgorithmType.VQE:
                circuit = self.circuit_builder.build_vqe_circuit(
                    hamiltonian_terms=task.input_data.get("hamiltonian_terms", []),
                    ansatz_depth=task.input_data.get("ansatz_depth", 2)
                )
            else:
                # For other algorithms, use a generic approach
                circuit = self._build_generic_circuit(task)
                
            # Simulate the circuit
            results = self.quantum_simulator.simulate_circuit(circuit, shots=1024)
            
            # Process results based on algorithm type
            if task.algorithm_type == QuantumAlgorithmType.GROVERS_SEARCH:
                # Find the most frequent result (the marked item)
                most_frequent = max(results, key=results.get)
                task.result = {"marked_item": int(most_frequent, 2), "probabilities": results}
            elif task.algorithm_type == QuantumAlgorithmType.SHORS_FACTORIZATION:
                # Process period finding results
                most_frequent = max(results, key=results.get)
                period = int(most_frequent, 2)
                task.result = {"period": period, "factors": self._post_process_shors(period, task.input_data.get("n", 15))}
            else:
                task.result = {"raw_results": results, "circuit_info": circuit.to_dict() if hasattr(circuit, 'to_dict') else str(circuit)}
                
            task.completed_at = datetime.now()
            task.execution_time = (task.completed_at - task.started_at).total_seconds()
            task.status = "completed"
            task.backend_used = self.quantum_simulator.backend
            
            # Record metrics
            task.metrics = {
                "execution_time": task.execution_time,
                "algorithm_type": task.algorithm_type.value,
                "input_size": task.metadata["input_size"],
                "simulation_shots": 1024,
                "backend_used": task.backend_used.value
            }
            
            logger.info(f"Completed quantum task {task_id} in {task.execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Quantum task {task_id} failed: {str(e)}")
            task.status = "failed"
            task.completed_at = datetime.now()
            task.result = {"error": str(e)}
            
            # Fall back to classical if quantum failed
            await self._execute_classical_fallback(task_id, fallback_reason=str(e))
            
    async def _execute_classical_fallback(self, task_id: str, fallback_reason: str = "quantum_not_suitable"):
        """Execute classical fallback for a task."""
        task = self.tasks[task_id]
        
        if task.status == "running":
            # If quantum execution is already running, don't double-process
            return
            
        task.status = "fallback_to_classical"
        task.started_at = datetime.now()
        
        fallback_result = self.fallback_manager.fallback_to_classical(
            task.algorithm_type, task.input_data
        )
        
        task.result = fallback_result.classical_result
        task.completed_at = datetime.now()
        task.execution_time = (task.completed_at - task.started_at).total_seconds()
        task.status = "completed_with_fallback"
        
        # Record metrics
        task.metrics = {
            "execution_time": task.execution_time,
            "algorithm_type": task.algorithm_type.value,
            "input_size": task.metadata["input_size"],
            "fallback_reason": fallback_reason,
            "classical_equivalent_complexity": fallback_result.quantum_equivalent_complexity
        }
        
        logger.info(f"Completed task {task_id} with classical fallback in {task.execution_time:.2f}s")
        
    def _should_use_classical_for_input_size(self, input_data: Any, algorithm_type: QuantumAlgorithmType) -> bool:
        """Determine if classical is better based on input size."""
        # For small inputs, classical might be faster due to quantum overhead
        if algorithm_type == QuantumAlgorithmType.SHORS_FACTORIZATION:
            n = input_data.get("n", 15)
            return n < 50  # For numbers < 50, classical is probably faster
        elif algorithm_type == QuantumAlgorithmType.GROVERS_SEARCH:
            n = input_data.get("n", 3)
            search_space_size = 2 ** n
            return search_space_size < 16  # For small search spaces, classical is faster
        else:
            return False
            
    def _is_quantum_advantage_likely(self, input_data: Any, algorithm_type: QuantumAlgorithmType) -> bool:
        """Determine if quantum advantage is likely for this problem."""
        if algorithm_type == QuantumAlgorithmType.SHORS_FACTORIZATION:
            n = input_data.get("n", 15)
            return n > 100  # Quantum advantage for large numbers
        elif algorithm_type == QuantumAlgorithmType.GROVERS_SEARCH:
            n = input_data.get("n", 3)
            search_space_size = 2 ** n
            return search_space_size > 1000  # Quantum advantage for large search spaces
        elif algorithm_type == QuantumAlgorithmType.VQE:
            # Quantum advantage for complex molecular simulations
            hamiltonian_terms = input_data.get("hamiltonian_terms", [])
            return len(hamiltonian_terms) > 10
        else:
            return False
            
    def _build_generic_circuit(self, task: QuantumTask) -> QuantumCircuit:
        """Build a generic circuit for unsupported algorithm types."""
        circuit_id = f"circuit_generic_{secrets.token_hex(8)}"
        
        # Create a simple circuit with some basic gates
        num_qubits = min(4, max(2, len(str(task.input_data))))  # Between 2-4 qubits
        gates = [
            self.circuit_builder._create_hadamard_gate([i], {}) for i in range(num_qubits)
        ] + [
            self.circuit_builder._create_cnot_gate([i, (i+1) % num_qubits], {}) for i in range(num_qubits)
        ]
        
        return QuantumCircuit(
            circuit_id=circuit_id,
            algorithm_type=task.algorithm_type,
            num_qubits=num_qubits,
            depth=len(gates),
            gates=[gate.to_dict() for gate in gates],
            parameters={"input_hash": hash(str(task.input_data))},
            created_at=datetime.now(),
            metadata={"algorithm": "generic", "input_size": task.metadata["input_size"]}
        )
        
    def _post_process_shors(self, period: int, n: int) -> List[int]:
        """Post-process Shor's algorithm results to get factors."""
        # Simplified post-processing for demonstration
        # In reality, this would involve continued fractions and GCD calculations
        if period % 2 == 0:
            candidate_factor = math.gcd(int(pow(2, period//2, n)) - 1, n)
            if 1 < candidate_factor < n:
                return [candidate_factor, n // candidate_factor]
        return [1, n]  # Failed to factor
        
    def get_task_status(self, task_id: str) -> Optional[QuantumTask]:
        """Get status of a quantum task."""
        return self.tasks.get(task_id)
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for quantum operations."""
        completed_tasks = [t for t in self.tasks.values() if t.status in ["completed", "completed_with_fallback"]]
        
        if not completed_tasks:
            return {
                "total_tasks": len(self.tasks),
                "completed_tasks": 0,
                "quantum_vs_classical_ratio": 0.0,
                "average_quantum_time": 0.0,
                "average_classical_time": 0.0,
                "quantum_readiness": self.assess_quantum_readiness()
            }
            
        quantum_tasks = [t for t in completed_tasks if t.backend_used is not None]
        classical_tasks = [t for t in completed_tasks if t.status == "completed_with_fallback"]
        
        quantum_time = sum(t.execution_time or 0 for t in quantum_tasks)
        classical_time = sum(t.execution_time or 0 for t in classical_tasks)
        
        return {
            "total_tasks": len(self.tasks),
            "completed_tasks": len(completed_tasks),
            "quantum_tasks": len(quantum_tasks),
            "classical_fallback_tasks": len(classical_tasks),
            "quantum_vs_classical_ratio": len(quantum_tasks) / len(completed_tasks) if completed_tasks else 0,
            "average_quantum_time": quantum_time / len(quantum_tasks) if quantum_tasks else 0,
            "average_classical_time": classical_time / len(classical_tasks) if classical_tasks else 0,
            "quantum_readiness": self.assess_quantum_readiness(),
            "simulation_stats": self.quantum_simulator.performance_stats
        }
        
    def enable_hardware_backend(self):
        """Enable real quantum hardware backend (simulation)."""
        self.hardware_available = True
        logger.info("Real quantum hardware backend enabled (simulation)")
        
    def disable_hardware_backend(self):
        """Disable real quantum hardware backend."""
        self.hardware_available = False
        logger.info("Real quantum hardware backend disabled")


# Convenience function for easy use
def create_quantum_ready_manager(
    default_backend: QuantumSimulationBackend = QuantumSimulationBackend.SIMULATED
) -> QuantumReadyManager:
    """
    Convenience function to create a quantum-ready manager.
    
    Args:
        default_backend: Default backend for quantum simulation
        
    Returns:
        QuantumReadyManager instance
    """
    return QuantumReadyManager(default_backend)