"""Phase 0 check: GPT-2's top 5 guesses for the next token, and their probabilities."""
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
model.eval()  # inference mode: no dropout, so output is repeatable

text = "747 is the Queen of the"
ids = tokenizer(text, return_tensors="pt").input_ids  # shape [1, number of tokens]

with torch.no_grad():
    logits = model(ids).logits[0, -1]   # raw scores for the next token, one per vocab entry
probs = torch.softmax(logits, dim=-1)   # 50,257 probabilities summing to 1

top_p, top_ids = torch.topk(probs, 5)
print(f"Context: {text!r}")
for r, (p, tid) in enumerate(zip(top_p.tolist(), top_ids.tolist())):
    print(f"r={r}  {tokenizer.decode([tid])!r:<12} {p*100:6.2f}%")
