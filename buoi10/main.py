# Buoi 10: Bigram Neural Net (Karpathy Lecture 2 Part 2)
# Verbatim tu karpathy/makemore - bigram neural net
# Chay: .\venv\Scripts\python.exe buoi10\main.py
import numpy as np
import os

words = open("buoi09/names.txt", 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# TODO A: khoi tao W, forward 1 cap
# Khởi tạo W(27x27) nhỏ
np.random.seed(0)
W = np.random.randn(27,27)/10

def forward(prev_ix):
    # prev_ix: 0..26 -> logits = W[prev]
    logits = W[prev_ix]
    #softmax 
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return probs

# Test 1 cặp: prev = '.' (0) -> next = 'e' (5) trong "emma"
prev, nxt = '.', 'e'
probs = forward(stoi[prev])
print(f"P(next='{nxt}'|prev='{prev}') = {probs[stoi[nxt]]:.4f}")
print(f"probs sum = {probs.sum():.4f} (phai = 1.0)")

# TODO B
# Tạo dataset list các cặp (prev, next)
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = np.array(xs)
ys = np.array(ys)
print(f"\nDataset: {len(xs)} cap, vi du xs[:5] = {xs[:5]}, ys[:5] = {ys[:5]}")

# Train loop
np.random.seed(1)
for step in range(1000):
    # forward cho batch nhỏ 32
    idx = np.random.randint(0, len(xs), 32)
    xb, yb = xs[idx], ys[idx]
    logits = W[xb]
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(32), yb] + 1e-9))
    # backward
    dlogits = probs.copy()
    dlogits[np.arange(32), yb] -= 1
    dlogits /= 32
    lr = 5
    for i, ix in enumerate(xb):
        W[ix] -= lr * dlogits[i]
    if step % 200 == 0:
        idx_eval = np.random.choice(len(xs), 1000)
        logits_eval = W[xs[idx_eval]]
        exp_eval = np.exp(logits_eval - np.max(logits_eval, axis=1, keepdims=True))
        probs_eval = exp_eval / np.sum(exp_eval, axis=1, keepdims=True)
        loss_eval = -np.mean(np.log(probs_eval[np.arange(1000), ys[idx_eval]] + 1e-9))
        print(f"step {step:4d} batch_loss {loss:.4f}  eval_loss {loss_eval:.4f}")

print("\n10 tên sinh từ Bigram NN (Sau train):")
for i in range(10):
    out = []
    ix = 0
    while True:
        logits = W[ix]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        ix = np.random.choice(27, p=probs)
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))

if __name__ == "__main__":
    print("Buoi 10 san sang - bat dau Phan A: W(27,27) + forward")
