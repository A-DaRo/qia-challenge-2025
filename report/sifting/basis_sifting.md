# 5.1 Basis Sifting Protocol

## Overview

Basis sifting discards measurement outcomes where Alice and Bob used different bases, retaining only matching-basis pairs for key generation.

## Sifting Algorithm

### Procedure

**Input**:
- $\mathcal{M}_A = \{(m_i^A, b_i^A)\}_{i=1}^n$ (Alice's outcomes and bases)
- $\mathcal{M}_B = \{(m_i^B, b_i^B)\}_{i=1}^n$ (Bob's outcomes and bases)

**Output**:
- $K_A = \{m_i^A : b_i^A = b_i^B\}$ (Alice's sifted key)
- $K_B = \{m_i^B : b_i^A = b_i^B\}$ (Bob's sifted key)

**Algorithm**:
```
K_A ← empty bitarray
K_B ← empty bitarray

for i = 1 to n:
    if b_i^A == b_i^B:
        K_A.append(m_i^A)
        K_B.append(m_i^B)

return (K_A, K_B)
```

### Implementation

```python
class Sifter:
    def sift(
        self,
        alice_outcomes: np.ndarray,
        alice_bases: np.ndarray,
        bob_outcomes: np.ndarray,
        bob_bases: np.ndarray,
    ) -> SiftingResult:
        # Find matching basis indices
        matching_mask = (alice_bases == bob_bases)
        matching_indices = np.where(matching_mask)[0]
        
        # Extract matching outcomes
        alice_sifted = bitarray(alice_outcomes[matching_indices].tolist())
        bob_sifted = bitarray(bob_outcomes[matching_indices].tolist())
        
        return SiftingResult(
            alice_sifted_key=alice_sifted,
            bob_sifted_key=bob_sifted,
            num_sifted_bits=len(alice_sifted),
        )
```

## Commit-Reveal Protocol

### Security Rationale

**Threat**: If Alice reveals her bases first, Bob can adaptively choose his bases to maximize information gain.

**Solution**: Commitment scheme prevents Alice from changing bases after Bob reveals his.

### Protocol Steps

1. **Commitment Phase**:
   ```python
   # Alice computes commitment
   salt = os.urandom(32)
   commitment = SHA256Commitment()
   commit_hash = commitment.commit(alice_bases, salt)
   
   # Alice → Bob: commit_hash
   yield from socket.send(MessageType.BASIS_COMMITMENT, {"hash": commit_hash})
   ```

2. **Revelation Phase**:
   ```python
   # Bob → Alice: bob_bases
   bob_msg = yield from socket.recv(MessageType.BASIS_REVELATION)
   bob_bases = bob_msg["bases"]
   
   # Alice → Bob: (alice_bases, salt)
   yield from socket.send(MessageType.BASIS_REVELATION, {
       "bases": alice_bases,
       "salt": salt,
   })
   ```

3. **Verification**:
   ```python
   # Bob verifies commitment
   alice_msg = yield from socket.recv(MessageType.BASIS_REVELATION)
   is_valid = commitment.verify(
       alice_msg["bases"],
       alice_msg["salt"],
       received_commit_hash,
   )
   
   if not is_valid:
       raise CommitmentViolationError("Basis commitment verification failed")
   ```

### SHA-256 Commitment

```python
class SHA256Commitment:
    def commit(self, bases: bitarray, salt: bytes) -> bytes:
        data = bases.tobytes() + salt
        return hashlib.sha256(data).digest()
    
    def verify(self, bases: bitarray, salt: bytes, commitment: bytes) -> bool:
        recomputed = self.commit(bases, salt)
        return recomputed == commitment
```

**Properties**:
- **Hiding**: Commitment reveals nothing about bases
- **Binding**: Alice cannot change bases after commitment

## Sifting Statistics

### Expected Sifting Rate

For uniform random bases:
$$
R_{\text{sift}} = P(b_A = b_B) = 0.5
$$

**Example**: $n_{\text{raw}} = 10,000 \Rightarrow n_{\text{sifted}} \approx 5,000$

### Variance (Binomial)

$$
\sigma^2 = n \cdot p \cdot (1 - p) = n \cdot 0.25
$$

**95% Confidence Interval**:
$$
n_{\text{sifted}} \in [0.5n - 1.96\sqrt{0.25n}, \, 0.5n + 1.96\sqrt{0.25n}]
$$

## References

- Bennett, C. H., & Brassard, G. (1984). "Quantum cryptography: Public key distribution and coin tossing." *Proc. IEEE Int. Conf. Computers, Systems and Signal Processing*, 175-179.
- Halevi, S., & Micali, S. (1996). "Practical and provably-secure commitment schemes from collision-free hashing." *CRYPTO'96*, 201-215.
- Caligo implementation: [`caligo/sifting/sifter.py`](../../caligo/caligo/sifting/sifter.py)
