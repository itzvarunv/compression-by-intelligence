"""Phase 2a - Varun's fixed-width scheme (kept as the documented dead end).

w = number of bits needed for the largest r.
Header: w ones, then a 0 (unary, so the decoder knows w).
Body:   every r written in exactly w bits.
Problem: one surprising token (large r) makes w big for EVERY token.
"""


def encode(ranks):
    w = max(ranks).bit_length() if ranks else 0
    header = "1" * w + "0"
    body = "".join(format(r, f"0{w}b") if w else "" for r in ranks)
    return header + body


def decode(bits, count):
    w = bits.index("0")          # count the leading ones
    pos = w + 1
    ranks = []
    for _ in range(count):
        ranks.append(int(bits[pos:pos + w], 2) if w else 0)
        pos += w
    return ranks


if __name__ == "__main__":
    import random
    random.seed(0)
    for _ in range(200):
        rs = [random.choice([0, 0, 0, 1, 2, 5, 40, 900, 30000]) for _ in range(random.randint(1, 50))]
        assert decode(encode(rs), len(rs)) == rs
    assert decode(encode([0, 0, 0]), 3) == [0, 0, 0]
    print("enc_fixed round-trip OK")
