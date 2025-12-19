
import logging
import numpy as np
from caligo.simulation.network_builder import perfect_network_config
from squidasm.run.stack.run import run as run_stack_network
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta

class AliceProgram(Program):
    PEER = "Bob"
    def __init__(self, num_pairs):
        self.num_pairs = num_pairs

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="Alice",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets[self.PEER]
        outcomes = []
        bases = []
        
        # Measure in Z basis (0) for first half, X basis (1) for second half
        for i in range(self.num_pairs):
            basis = 0 if i < self.num_pairs // 2 else 1
            q = epr_socket.create_keep(1)[0]
            if basis == 1:
                q.H()
            m = q.measure()
            yield from context.connection.flush()
            outcomes.append(int(m))
            bases.append(basis)
            
        return {"outcomes": outcomes, "bases": bases}

class BobProgram(Program):
    PEER = "Alice"
    def __init__(self, num_pairs):
        self.num_pairs = num_pairs

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="Bob",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets[self.PEER]
        outcomes = []
        bases = []
        
        for i in range(self.num_pairs):
            basis = 0 if i < self.num_pairs // 2 else 1
            q = epr_socket.recv_keep(1)[0]
            if basis == 1:
                q.H()
            m = q.measure()
            yield from context.connection.flush()
            outcomes.append(int(m))
            bases.append(basis)
            
        return {"outcomes": outcomes, "bases": bases}

def run_test():
    logging.basicConfig(level=logging.INFO)
    network_cfg = perfect_network_config(num_qubits=10)
    
    num_pairs = 1000
    alice = AliceProgram(num_pairs)
    bob = BobProgram(num_pairs)
    
    results = run_stack_network(network_cfg, {"Alice": alice, "Bob": bob})
    
    # Process results
    alice_res = results[0][0]
    bob_res = results[1][0]
    
    alice_out = np.array(alice_res["outcomes"])
    bob_out = np.array(bob_res["outcomes"])
    
    # Check Z basis correlation (first half)
    z_slice = slice(0, num_pairs // 2)
    z_errors = np.sum(alice_out[z_slice] != bob_out[z_slice])
    z_qber = z_errors / (num_pairs // 2)
    print(f"Z Basis QBER: {z_qber}")
    
    # Check X basis correlation (second half)
    x_slice = slice(num_pairs // 2, num_pairs)
    x_errors = np.sum(alice_out[x_slice] != bob_out[x_slice])
    x_qber = x_errors / (num_pairs // 2)
    print(f"X Basis QBER: {x_qber}")

if __name__ == "__main__":
    run_test()
