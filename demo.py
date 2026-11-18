"""Interactive demo: compress what you type, save it as a .bin file, decode it back.

  python demo.py              full walk-through: seed -> text -> compress -> .bin -> decode -> compare
  python demo.py d file.bin   decode an existing .bin file (compares with the original if it was saved)

The seed is stored in the file as plain text (UTF-8). GPT-2 reads the seed first, so it
already knows the topic when your text starts, and predicts it better.
The decoder reads the seed from the file, feeds it to GPT-2 the same way, then rebuilds
the text token by token.

File format:
  byte 0          encoder id (0 = gamma, 1 = arith)
  bytes 1-2       seed length in bytes
  next L bytes    seed text
  next 4 bytes    text token count
  next 1 byte     padding bits
  rest            compressed text bits
"""
import gzip
import json
import math
import sys
from rank_stream import Predictor
import enc_gamma, enc_arith
from compress import pack, unpack

ENCODERS = ["gamma", "arith"]


def start(pred, seed):
    """Fresh context (EOT), then feed every seed token into GPT-2."""
    pred.reset()
    for t in pred.to_ids(seed):
        pred.push(t)


def encode(seed, para, enc_name, pred):
    """-> (file bytes, per-token rows of (token text, r, p))"""
    start(pred, seed)
    ids = pred.to_ids(para)
    ranks, rows = [], []
    arith = enc_arith.Encoder()
    for t in ids:
        r = int((pred.ranked() == t).nonzero().item())
        probs = pred.probs()
        if enc_name == "arith":
            cum = enc_arith.cum_freqs(probs)
            arith.encode(int(cum[t]), int(cum[t + 1]), int(cum[-1]))
        ranks.append(r)
        rows.append((pred.to_text([t]), r, float(probs[t])))
        pred.push(t)
    bits = arith.finish() if enc_name == "arith" else enc_gamma.encode(ranks)
    seed_b = seed.encode("utf-8")
    pad, payload = pack(bits)
    blob = (bytes([ENCODERS.index(enc_name)]) + len(seed_b).to_bytes(2, "big") + seed_b
            + len(ids).to_bytes(4, "big") + bytes([pad]) + payload)
    return blob, rows


def decode(blob, pred):
    """-> (seed, paragraph)"""
    enc_name = ENCODERS[blob[0]]
    L = int.from_bytes(blob[1:3], "big")
    seed = blob[3:3 + L].decode("utf-8")
    i = 3 + L
    count = int.from_bytes(blob[i:i + 4], "big")
    bits = unpack(blob[i + 5:], blob[i + 4])
    start(pred, seed)
    ids = []
    if enc_name == "arith":
        dec = enc_arith.Decoder(bits)
        for _ in range(count):
            t = dec.decode(enc_arith.cum_freqs(pred.probs()))
            ids.append(t)
            pred.push(t)
    else:
        for r in enc_gamma.decode(bits, count):
            t = int(pred.ranked()[r])
            ids.append(t)
            pred.push(t)
    return seed, pred.to_text(ids)


LINE = "=" * 64


def ask(prompt, default=""):
    try:
        ans = input(prompt).strip()
    except EOFError:
        ans = ""
    return ans or default


def read_text():
    lines = []
    while True:
        try:
            line = input("  > ")
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def header_bytes(seed):
    return 8 + len(seed.encode("utf-8"))   # encoder id + seed length + seed + token count + padding


def show_table(rows):
    print(f"\n  {'token':<16}{'r':>7}{'p %':>9}{'bits':>7}")
    for tok, r, p in rows:
        print(f"  {tok!r:<16}{r:>7}{100 * p:>9.3f}{-math.log2(p):>7.1f}")
    print("  r = wrong guesses before the hit, p = GPT-2's probability, bits = -log2 p")


def compare(seed, text, orig_seed, orig_text):
    print("\n" + LINE + "\nCOMPARE with what you entered\n" + LINE)
    if (seed, text) == (orig_seed, orig_text):
        print(f"  MATCH: all {len(orig_seed) + len(orig_text)} characters identical (lossless)")
        return
    a, b = orig_seed + "\n" + orig_text, seed + "\n" + text
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    print(f"  MISMATCH at character {i}: expected {a[i:i + 20]!r}, got {b[i:i + 20]!r}")


def summary(seed, text, blob, enc_name, no_seed_blob):
    seed_b, text_b = len(seed.encode("utf-8")), len(text.encode("utf-8"))
    comp = len(blob) - header_bytes(seed)
    gz = len(gzip.compress(text.encode("utf-8"), 9))
    print("\n" + LINE + "\nSUMMARY\n" + LINE)
    print(f"  encoder                      : {enc_name}")
    print(f"  you typed (seed + text)      : {seed_b + text_b} bytes")
    print(f"  .bin file                    : {len(blob)} bytes  ({100 * (1 - len(blob) / (seed_b + text_b)):.0f}% smaller)")
    print(f"     seed, stored as plain text: {seed_b} bytes")
    print(f"     header                    : 8 bytes")
    print(f"     your text, compressed     : {comp} bytes")
    print(f"  your text alone              : {text_b} -> {comp} bytes  "
          f"({8 * comp / max(len(text), 1):.2f} bits/char, {text_b / max(comp, 1):.1f}x smaller)")
    print(f"  same text with NO seed       : {len(no_seed_blob) - 8} bytes")
    print(f"  same text with gzip          : {gz} bytes")


def run_decode(path, pred, orig=None):
    print("\n" + LINE + "\nSTEP 4  Decode\n" + LINE)
    with open(path, "rb") as f:
        blob = f.read()
    print(f"  reading {path} ({len(blob)} bytes), GPT-2 is rebuilding your text ...")
    seed, text = decode(blob, pred)
    out = path.rsplit(".", 1)[0] + "_out.txt"
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"\n  seed : {seed}")
    print(f"  text : {text}")
    print(f"  (text saved to {out})")
    if orig is None:
        try:
            with open(path.rsplit(".", 1)[0] + "_original.json", encoding="utf-8") as f:
                orig = json.load(f)
        except FileNotFoundError:
            print("\n  (no original saved for this file, so nothing to compare against)")
    if orig is not None:
        compare(seed, text, orig["seed"], orig["text"])
    return seed, text, blob


def walkthrough(pred):
    print(LINE)
    print("COMPRESSION BY INTELLIGENCE: demo")
    print(LINE)
    print("GPT-2 reads your seed, then tries to guess your text one token at a time.")
    print("We only store HOW WRONG its guesses were. Good guesses cost almost nothing,")
    print("so text the model finds predictable shrinks a lot. The decoder runs the same")
    print("model, replays the same guesses, and gets your exact text back.\n")

    print(LINE + "\nSTEP 1  Enter a seed text (the context GPT-2 reads first)\n" + LINE)
    seed = ask("  > ")
    print("\n" + LINE + "\nSTEP 2  Now write whatever comes after the seed\n        (press Enter on an empty line to finish)\n" + LINE)
    text = read_text()
    enc_name = ask("\n  Encoder: arith or gamma? [arith] ", "arith")
    if enc_name not in ENCODERS:
        enc_name = "arith"

    print("\n" + LINE + "\nSTEP 3  Compress\n" + LINE)
    print("  GPT-2 is guessing your text ...")
    blob, rows = encode(seed, text, enc_name, pred)
    payload = blob[header_bytes(seed):]
    bits = "".join(format(b, "08b") for b in payload)
    print(f"\n  Here is your compressed text ({len(payload)} bytes):")
    print(f"  bits: {bits[:96]}{' ...' if len(bits) > 96 else ''}")
    print(f"  hex : {payload.hex(' ')[:96]}{' ...' if len(payload) > 32 else ''}")

    name = ask("\n  Save as [demo.bin]: ", "demo.bin")
    if not name.endswith(".bin"):
        name += ".bin"
    with open(name, "wb") as f:
        f.write(blob)
    with open(name.rsplit(".", 1)[0] + "_original.json", "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "text": text}, f, ensure_ascii=False)
    print(f"  saved {name} ({len(blob)} bytes)")

    if ask("\n  Show token-by-token table? [y/N] ", "n").lower().startswith("y"):
        show_table(rows)

    path = ask(f"\n  Enter the .bin file name to decode [{name}]: ", name)
    run_decode(path, pred, orig={"seed": seed, "text": text})

    no_seed_blob, _ = encode("", text, enc_name, pred)
    summary(seed, text, blob, enc_name, no_seed_blob)


if __name__ == "__main__":
    pred = Predictor()
    if len(sys.argv) > 2 and sys.argv[1] == "d":
        seed, text, blob = run_decode(sys.argv[2], pred)
    else:
        walkthrough(pred)
