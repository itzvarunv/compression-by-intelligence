"""Phase 2b - Elias gamma: a length tag on EVERY number.

For each r:
  n = r + 1            (gamma can't write 0)
  k = number of binary digits of n
  write k-1 zeros, then n in binary.
r=0 -> '1' (1 bit), r=1 -> '010', r=2 -> '011', r=20000 -> 29 bits.
The decoder counts zeros to learn k, then reads k digits.
"""


def encode(ranks):
    out = []
    for r in ranks:
        b = format(r + 1, "b")
        out.append("0" * (len(b) - 1) + b)
    return "".join(out)


def decode(bits, count):
    ranks, pos = [], 0
    for _ in range(count):
        z = 0
        while bits[pos] == "0":
            z += 1
            pos += 1
        n = int(bits[pos:pos + z + 1], 2)
        pos += z + 1
        ranks.append(n - 1)
    return ranks


def cost(r):
    """Bits gamma spends on one r: 2*floor(log2(n)) + 1."""
    return 2 * ((r + 1).bit_length() - 1) + 1


if __name__ == "__main__":
    import random
    random.seed(0)
    assert encode([0, 1, 2]) == "1" + "010" + "011"
    for _ in range(200):
        rs = [random.choice([0, 0, 0, 1, 2, 5, 40, 900, 30000]) for _ in range(random.randint(1, 50))]
        assert decode(encode(rs), len(rs)) == rs
    print("enc_gamma round-trip OK")
