"""Phase 4 - Experiments.

For every file in texts/: bits per character (bpc) for raw, gzip, xz, and the three
encoders, time per token, and a lossless check. Saves:
  assets/results.md   the table
  assets/bpc.png      bpc per method, one panel per text
  assets/rank_hist.png  how often each size of r occurs (all texts together)
"""
import glob, gzip, json, lzma, math, os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rank_stream import Predictor, encode_ranks, decode_ranks
import enc_fixed, enc_gamma, enc_arith
from compress import compress, decompress

HEADER_BYTES = 6   # our file header (encoder id, token count, padding)


def size_bytes(bits):
    return HEADER_BYTES + math.ceil(len(bits) / 8)


def evaluate(path, pred):
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    raw = text.encode("utf-8")
    chars = len(text)
    res = {"name": os.path.basename(path), "chars": chars}

    t0 = time.perf_counter()
    ids, ranks, ps = encode_ranks(text, pred)          # one model pass gives r for fixed AND gamma
    t_rank_enc = time.perf_counter() - t0
    n = len(ranks)
    res["tokens"] = n

    bf, bg = enc_fixed.encode(ranks), enc_gamma.encode(ranks)
    assert enc_fixed.decode(bf, n) == ranks

    t0 = time.perf_counter()
    blob_a = compress(text, "arith", pred)
    t_arith_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    ok_rank = decode_ranks(enc_gamma.decode(bg, n), pred) == text
    t_rank_dec = time.perf_counter() - t0

    t0 = time.perf_counter()
    ok_arith = decompress(blob_a, pred) == text
    t_arith_dec = time.perf_counter() - t0

    sizes = {
        "raw": len(raw),
        "gzip": len(gzip.compress(raw, 9)),
        "xz": len(lzma.compress(raw, preset=9)),
        "fixed": size_bytes(bf),
        "gamma": size_bytes(bg),
        "arith": len(blob_a),
    }
    res["bpc"] = {k: 8 * v / chars for k, v in sizes.items()}
    res["ideal"] = sum(-math.log2(p) for p in ps) / chars   # -log2 p total: arith's target
    res["ok"] = ok_rank and ok_arith
    res["ms_per_token"] = {
        "enc_rank": 1000 * t_rank_enc / n, "enc_arith": 1000 * t_arith_enc / n,
        "dec_rank": 1000 * t_rank_dec / n, "dec_arith": 1000 * t_arith_dec / n,
    }
    res["ranks"] = ranks
    res["w"] = max(ranks).bit_length()
    return res


def table(results):
    m = ["raw", "gzip", "xz", "fixed", "gamma", "arith"]
    lines = ["| text | chars | tokens | " + " | ".join(m) + " | ideal | w | lossless |",
             "|" + "---|" * (len(m) + 6)]
    for r in results:
        lines.append(f"| {r['name']} | {r['chars']} | {r['tokens']} | "
                     + " | ".join(f"{r['bpc'][k]:.2f}" for k in m)
                     + f" | {r['ideal']:.2f} | {r['w']} | {'yes' if r['ok'] else 'NO'} |")
    lines += ["", "Values are bits per character (raw ASCII = 8). 'ideal' = sum of -log2 p / chars. "
              "'w' = fixed-width bits per token, set by the largest r.", "",
              "| text | enc rank ms/token | enc arith ms/token | dec rank ms/token | dec arith ms/token |",
              "|---|---|---|---|---|"]
    for r in results:
        t = r["ms_per_token"]
        lines.append(f"| {r['name']} | {t['enc_rank']:.1f} | {t['enc_arith']:.1f} | {t['dec_rank']:.1f} | {t['dec_arith']:.1f} |")
    return "\n".join(lines)


SURFACE, INK, INK2, MUTED, ACCENT = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7", "#2a78d6"


def plot_bpc(results, path):
    m = ["gzip", "xz", "fixed", "gamma", "arith"]
    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.4), sharex=True, facecolor=SURFACE)
    plt.setp(axes, xlim=(0, 10))
    axes = [axes] if len(results) == 1 else axes
    for ax, r in zip(axes, results):
        ax.set_facecolor(SURFACE)
        vals = [r["bpc"][k] for k in m]
        ax.barh(m, vals, color=[ACCENT if k == "arith" else MUTED for k in m], height=0.6)
        ax.axvline(8, color=MUTED, lw=1, ls="--", zorder=0)
        ax.text(8, -0.75, "raw = 8", fontsize=7, color=INK2, ha="center")
        for y, v in enumerate(vals):
            ax.text(v + 0.15, y, f"{v:.2f}", va="center", fontsize=8, color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
        ax.set_title(r["name"], fontsize=10, color=INK, loc="left")
        ax.invert_yaxis()
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(MUTED)
    fig.supxlabel("bits per character (lower is better)", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_ranks(results, path):
    allr = [x for r in results for x in r["ranks"]]
    # bins: r = 0, 1, 2-3, 4-7, ... (each bin = one extra gamma length step)
    k = [(x + 1).bit_length() - 1 for x in allr]
    nb = max(k) + 1
    counts = [k.count(i) for i in range(nb)]
    labels = ["0"] + [f"{2**i - 1}-{2**(i+1) - 2}" if i > 1 else "1-2" for i in range(1, nb)]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(range(nb), counts, color=ACCENT, width=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(nb))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color=INK2)
    ax.set_xlabel("r (number of wrong guesses before the hit)", fontsize=9, color=INK2)
    ax.set_ylabel("tokens (log scale)", fontsize=9, color=INK2)
    share0 = counts[0] / len(allr)
    ax.set_title(f"How surprised was GPT-2? {share0:.0%} of {len(allr)} tokens were its first guess",
                 fontsize=10, color=INK, loc="left")
    ax.tick_params(colors=INK2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    # python eval.py            -> evaluate every file in texts/
    # python eval.py a.txt ...  -> evaluate only these, keep earlier results for the rest
    os.makedirs("assets", exist_ok=True)
    store = "assets/results.json"
    saved = json.load(open(store)) if os.path.exists(store) else {}
    paths = sys.argv[1:] or sorted(glob.glob("texts/*.txt"))
    pred = Predictor()
    for path in paths:
        print("evaluating", path, flush=True)
        r = evaluate(path, pred)
        saved[r["name"]] = r
        json.dump(saved, open(store, "w"))
    results = [saved[k] for k in sorted(saved)]
    md = table(results)
    print(md)
    with open("assets/results.md", "w") as f:
        f.write(md + "\n")
    plot_bpc(results, "assets/bpc.png")
    plot_ranks(results, "assets/rank_hist.png")
    print("saved assets/results.md, assets/bpc.png, assets/rank_hist.png")
