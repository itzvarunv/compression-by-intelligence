# Compression is Intelligence: Action Plan

**Goal:** a lossless text compressor that uses GPT-2's ranked next-token predictions.
We compare three ways of encoding the predictions, and benchmark them against gzip and xz.

## Ground rules

- **Claude writes the code and runs each phase straight through, then walks Varun through it.**
- This plan is followed strictly. If any step needs to change, Claude stops and asks Varun first.
- Old and dead-end versions stay in the repo (e.g. the fixed-width encoder) so readers can see why they failed.

## Symbols used throughout

| Symbol | Meaning |
|---|---|
| t | a token (a word or part of a word, as GPT-2's tokenizer splits text) |
| r | rank of the actual token in the model's list = number of wrong guesses before the hit (0 = first guess right) |
| p | probability the model gave the actual token |
| n | r + 1 (used by Elias gamma, which can't encode 0) |
| k | number of binary digits of n |
| w | fixed bit width in the fixed-width scheme = bits needed for the largest r |
| bpc | bits per character of the original text (raw ASCII = 8) |

## Phase 0: Setup (on the Mac)

- [ ] Create a venv in `Updated/`: `python3 -m venv .venv && source .venv/bin/activate`
- [ ] Install: `pip install torch transformers matplotlib`
- [ ] Download GPT-2 small (124M params, ~500 MB) with a one-line test script that prints the top 5 next tokens for "747 is the Queen of the"
- [ ] Settings for determinism: CPU only, float32, `model.eval()`, `torch.no_grad()`, no sampling

**Checkpoint:** top-5 list prints, and gives the same output on two runs.

## Phase 1: Step 1, the rank stream (`rank_stream.py`)

- [ ] `encode_ranks(text)` → tokenize, then for each token feed the context so far into GPT-2, sort the predictions, and record r and p
- [ ] The first token has no context: start from GPT-2's `<|endoftext|>` token (so no separate seed is needed)
- [ ] `decode_ranks(ranks)` → rebuild the text using only the r list and the same model
- [ ] Unary bitstream view: r zeros then a 1 (your notebook scheme), only for display and size comparison

**Checkpoint:** `decode_ranks(encode_ranks(text)) == text` on 3 different texts.

## Phase 2: Step 2, the three encoders

Each file has `encode(list) -> bits` and `decode(bits) -> list`, plus a round-trip test.

- [ ] `enc_fixed.py`: Varun's scheme. Unary header for w (w ones, then a 0), then every r in w bits
- [ ] `enc_gamma.py`: Elias gamma. For each r: n = r + 1, write k − 1 zeros, then n in binary
- [ ] `enc_arith.py`: arithmetic coding using p (cost ≈ −log2 p bits per token)

**Checkpoint:** all three round-trip exactly, and the bit counts on the "747 … joker" sentence match a hand calculation.

## Phase 3: Full compressor (`compress.py`)

- [ ] `python compress.py c input.txt out.bin` and `python compress.py d out.bin back.txt`, with a flag to choose the encoder
- [ ] Pack bits into bytes and store the number of padding bits

**Checkpoint:** `back.txt` is byte-identical to `input.txt`.

## Phase 4: Experiments (`eval.py`)

- [ ] Test texts in `texts/`: a Wikipedia-style paragraph, a lecture transcript, Python code, random gibberish
- [ ] Table of bpc for: raw (8), gzip, xz, fixed, gamma, arithmetic
- [ ] Time per token (compress and decompress)
- [ ] Histogram of r (log scale), showing the rare high-rank "joker" outliers

**Checkpoint:** table and plots saved to `assets/`.

## Phase 5: Repo and README

- [ ] README: idea, flowchart of the pipeline, results table, screenshots/plots from `assets/`
- [ ] Section on why fixed width fails (one outlier sets w for every token)
- [ ] Push to GitHub

## Layout

```
Updated/
├── action_plan.md
├── rank_stream.py
├── enc_fixed.py  enc_gamma.py  enc_arith.py
├── compress.py
├── eval.py
├── texts/
└── assets/
```
