"""Phase 3 - Full compressor.

  python compress.py c input.txt out.bin [fixed|gamma|arith]   (default: arith)
  python compress.py d out.bin back.txt

File format:
  byte 0     encoder id (0 = fixed, 1 = gamma, 2 = arith)
  bytes 1-4  token count (needed so the decoder knows when to stop)
  byte 5     padding bits added to fill the last byte
  rest       the bitstream, packed 8 bits per byte
"""
import sys
from rank_stream import Predictor, encode_ranks, decode_ranks
import enc_fixed, enc_gamma, enc_arith

ENCODERS = ["fixed", "gamma", "arith"]


def pack(bits):
    pad = (-len(bits)) % 8
    bits += "0" * pad
    return pad, bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def unpack(data, pad):
    bits = "".join(format(b, "08b") for b in data)
    return bits[:len(bits) - pad] if pad else bits


def compress(text, enc_name, pred):
    if enc_name == "arith":
        bits, count = enc_arith.encode_text(text, pred)
    else:
        ids, ranks, _ = encode_ranks(text, pred)
        bits = (enc_fixed if enc_name == "fixed" else enc_gamma).encode(ranks)
        count = len(ranks)
    pad, payload = pack(bits)
    header = bytes([ENCODERS.index(enc_name)]) + count.to_bytes(4, "big") + bytes([pad])
    return header + payload


def decompress(blob, pred):
    enc_name = ENCODERS[blob[0]]
    count = int.from_bytes(blob[1:5], "big")
    bits = unpack(blob[6:], blob[5])
    if count == 0:
        return ""
    if enc_name == "arith":
        return enc_arith.decode_text(bits, count, pred)
    ranks = (enc_fixed if enc_name == "fixed" else enc_gamma).decode(bits, count)
    return decode_ranks(ranks, pred)


if __name__ == "__main__":
    mode, src, dst = sys.argv[1:4]
    pred = Predictor()
    if mode == "c":
        enc_name = sys.argv[4] if len(sys.argv) > 4 else "arith"
        with open(src, encoding="utf-8", newline="") as f:
            text = f.read()
        blob = compress(text, enc_name, pred)
        with open(dst, "wb") as f:
            f.write(blob)
        n_in = len(text.encode("utf-8"))
        print(f"{enc_name}: {n_in} bytes -> {len(blob)} bytes  ({8 * len(blob) / max(len(text), 1):.3f} bits/char)")
    elif mode == "d":
        with open(src, "rb") as f:
            text = decompress(f.read(), pred)
        with open(dst, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(f"decompressed -> {dst}")
