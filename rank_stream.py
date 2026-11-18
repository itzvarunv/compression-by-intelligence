"""Phase 1 - Step 1: the rank stream.

For every token t in the text, GPT-2 ranks all 50,257 possible next tokens.
r = position of the actual token in that ranked list
  = number of wrong guesses before the hit (0 = first guess right).
p = probability the model gave the actual token.

The decoder rebuilds the text from the r list alone, by running the same model
and picking the r-th guess at every step.
"""
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

EOT = 50256    # GPT-2's <|endoftext|> token: the empty starting context (no seed needed)
CHUNK = 1000   # tokens per context window; GPT-2 can only see 1024, so we restart from EOT


class Predictor:
    """Wraps GPT-2 so encoder and decoder walk through the text in exactly the same way."""

    def __init__(self):
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
        self.model.eval()

    def reset(self):
        """Start a fresh context containing only EOT."""
        self.past = None   # key/value cache: lets the model reuse earlier work, one token at a time
        self.n = 0         # tokens in the current context
        self.logits = self._feed(EOT)

    def _feed(self, tid):
        with torch.no_grad():
            out = self.model(torch.tensor([[tid]]), past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        self.n += 1
        return out.logits[0, -1]   # scores for the NEXT token

    def push(self, tid):
        """Add the real token to the context and compute the next prediction."""
        if self.n >= CHUNK:
            self.reset()
        else:
            self.logits = self._feed(tid)

    def ranked(self):
        """All token ids, best guess first. stable=True so ties always break the same way."""
        return torch.argsort(self.logits, descending=True, stable=True)

    def probs(self):
        return torch.softmax(self.logits, dim=-1)

    def to_ids(self, text):
        return self.tok.encode(text)

    def to_text(self, ids):
        return self.tok.decode(ids, clean_up_tokenization_spaces=False)


def encode_ranks(text, pred):
    """text -> (token ids, r list, p list)"""
    ids = pred.to_ids(text)
    pred.reset()
    ranks, ps = [], []
    for t in ids:
        order = pred.ranked()
        ranks.append(int((order == t).nonzero().item()))
        ps.append(float(pred.probs()[t]))
        pred.push(t)
    return ids, ranks, ps


def decode_ranks(ranks, pred):
    """r list -> text"""
    pred.reset()
    ids = []
    for r in ranks:
        t = int(pred.ranked()[r])
        ids.append(t)
        pred.push(t)
    return pred.to_text(ids)


def unary(ranks):
    """Varun's notebook view: r zeros then a 1, for each token."""
    return "".join("0" * r + "1" for r in ranks)


if __name__ == "__main__":
    pred = Predictor()
    tests = [
        "747 is the Queen of the skies but A380 is a joker",
        "The quick brown fox jumps over the lazy dog.\nNew line, tabs\tand  double  spaces!",
        "def f(x):\n    return x ** 2  # Python code with symbols {}[]<>@#$%",
    ]
    for text in tests:
        ids, ranks, ps = encode_ranks(text, pred)
        back = decode_ranks(ranks, pred)
        print(f"{text!r}")
        print("  tokens:", [pred.to_text([i]) for i in ids])
        print("  r     :", ranks)
        print("  unary bits:", len(unary(ranks)), "| round-trip OK:", back == text)
