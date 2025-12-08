"""
Quick test to verify EPR noise is working.

This minimal test generates EPR pairs with a noisy channel and checks
if we observe the expected error rate.
"""
import numpy as np
from netqasm.sdk import EPRSocket, Qubit
from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta

class AliceTestProgram(Program):
    """Alice generates EPR pairs and measures in Z basis."""
    
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="alice_test",
            csockets=["bob"],
            epr_sockets=["bob"],
            max_qubits=2,
        )
    
    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets["bob"]
        connection = context.connection
        csocket = context.csockets["bob"]
        
        num_pairs = 1000
        outcomes_alice = []
        
        for _ in range(num_pairs):
            # Create EPR pair
            q = epr_socket.create_keep()[0]
            
            # Measure in Z basis (no rotation)
            m = q.measure()
            yield from connection.flush()
            outcomes_alice.append(int(m))
        
        # Send outcomes to Bob
        for outcome in outcomes_alice:
            csocket.send(str(outcome))
        
        return {"outcomes": outcomes_alice}


class BobTestProgram(Program):
    """Bob receives EPR pairs and measures in Z basis."""
    
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="bob_test",
            csockets=["alice"],
            epr_sockets=["alice"],
            max_qubits=2,
        )
    
    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets["alice"]
        connection = context.connection
        csocket = context.csockets["alice"]
        
        num_pairs = 1000
        outcomes_bob = []
        
        for _ in range(num_pairs):
            # Receive EPR pair
            q = epr_socket.recv_keep()[0]
            
            # Measure in Z basis (no rotation)
            m = q.measure()
            yield from connection.flush()
            outcomes_bob.append(int(m))
        
        # Receive Alice's outcomes
        outcomes_alice = []
        for _ in range(num_pairs):
            msg = yield from csocket.recv()
            outcomes_alice.append(int(msg))
        
        # Calculate error rate
        outcomes_alice = np.array(outcomes_alice)
        outcomes_bob = np.array(outcomes_bob)
        errors = np.sum(outcomes_alice != outcomes_bob)
        error_rate = errors / num_pairs
        
        print(f"\n{'='*60}")
        print(f"EPR Noise Test Results")
        print(f"{'='*60}")
        print(f"Total pairs: {num_pairs}")
        print(f"Errors: {errors}")
        print(f"Error rate: {error_rate*100:.2f}%")
        print(f"Expected (fidelity=0.97): ~2.25%")
        print(f"{'='*60}\n")
        
        return {"outcomes": outcomes_bob, "error_rate": error_rate}


if __name__ == "__main__":
    cfg = StackNetworkConfig.from_file("ehok/configs/network_baseline.yaml")
    
    alice_program = AliceTestProgram()
    bob_program = BobTestProgram()
    
    alice_results, bob_results = run(
        config=cfg,
        programs={"alice": alice_program, "bob": bob_program},
        num_times=1,
    )
    
    print(f"Test complete. Error rate: {bob_results[0]['error_rate']*100:.2f}%")
