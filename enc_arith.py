"""Phase 2c - Arithmetic coding on p (the model's probabilities).

Idea: the whole message becomes one number inside [0, 1). Each token shrinks the
current interval to the slice its probability owns; a token with probability p
costs about -log2(p) bits. Very likely tokens cost a fraction of a bit.

Unlike fixed/gamma, this needs the model's FULL distribution at every step,
not just r, so the decoder must run GPT-2 in the loop (encode_text / decode_text).

Integer version (Witten-Neal-Cleary, 32-bit), so encoder and decoder do
exactly the same arithmetic.
"""
import torch

BITS = 32
FULL = (1 << BITS) - 1
HALF = 1 << (BITS - 1)
QUART = 1 << (BITS - 2)
SCALE = 1 << 24   # probabilities -> integer counts (each token gets at least 1)


def cum_freqs(probs):
    """Cumulative integer counts: token s owns [cum[s], cum[s+1]). Total = cum[-1]."""
    f = (probs * SCALE).floor().long() + 1
    return torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(f, 0)])


class Encoder:
    def __init__(self):
        self.low, self.high, self.pending, self.out = 0, FULL, 0, []

    def _emit(self, bit):
        self.out.append(bit)
        self.out.extend([1 - bit] * self.pending)
        self.pending = 0

    def encode(self, lo, hi, total):
        rng = self.high - self.low + 1
        self.high = self.low + rng * hi // total - 1
        self.low = self.low + rng * lo // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUART and self.high < 3 * QUART:
                self.pending += 1
                self.low -= QUART
                self.high -= QUART
            else:
                break
            self.low = 2 * self.low
            self.high = 2 * self.high + 1

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < QUART else 1)
        return "".join(map(str, self.out))


class Decoder:
    def __init__(self, bits):
        self.bits, self.pos = bits, 0
        self.low, self.high, self.value = 0, FULL, 0
        for _ in range(BITS):
            self.value = 2 * self.value + self._next()

    def _next(self):
        b = int(self.bits[self.pos]) if self.pos < len(self.bits) else 0
        self.pos += 1
        return b

    def decode(self, cum):
        total = int(cum[-1])
        rng = self.high - self.low + 1
        target = ((self.value - self.low + 1) * total - 1) // rng
        s = int(torch.searchsorted(cum, torch.tensor(target), right=True)) - 1
        lo, hi = int(cum[s]), int(cum[s + 1])
        self.high = self.low + rng * hi // total - 1
        self.low = self.low + rng * lo // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUART and self.high < 3 * QUART:
                self.low -= QUART
                self.high -= QUART
                self.value -= QUART
            else:
                break
            self.low = 2 * self.low
            self.high = 2 * self.high + 1
            self.value = 2 * self.value + self._next()
        return s


def encode_text(text, pred):
    """text -> (bit string, token count), using GPT-2 (a rank_stream.Predictor)."""
    ids = pred.to_ids(text)
    pred.reset()
    enc = Encoder()
    for t in ids:
        cum = cum_freqs(pred.probs())
        enc.encode(int(cum[t]), int(cum[t + 1]), int(cum[-1]))
        pred.push(t)
    return enc.finish(), len(ids)


def decode_text(bits, count, pred):
    pred.reset()
    dec = Decoder(bits)
    ids = []
    for _ in range(count):
        t = dec.decode(cum_freqs(pred.probs()))
        ids.append(t)
        pred.push(t)
    return pred.to_text(ids)


if __name__ == "__main__":
    # Coder self-test with a fixed toy distribution (no GPT-2 needed)
    import random
    random.seed(0)
    probs = torch.tensor([0.6, 0.25, 0.1, 0.04, 0.01])
    cum = cum_freqs(probs)
    for _ in range(50):
        msg = random.choices(range(5), weights=probs.tolist(), k=random.randint(1, 300))
        e = Encoder()
        for s in msg:
            e.encode(int(cum[s]), int(cum[s + 1]), int(cum[-1]))
        bits = e.finish()
        d = Decoder(bits)
        assert [d.decode(cum) for _ in msg] == msg
    print("enc_arith coder round-trip OK")
