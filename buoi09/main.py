# Buoi 9: makemore Bigram Language Model (Karpathy Lecture 2)
# Verbatim tu karpathy/makemore/bigram.py — se hoc chi tiet mai
# Chay: .\venv\Scripts\python.exe buoi09\main.py
# Link goc: https://github.com/karpathy/makemore/blob/master/bigram.py
import os
import numpy as np
import urllib.request

# TODO Buoi 9:
# Phan A: doc names.txt, stoi/itos, dem Bigram C (27x27)
# Tải names.txt
if not os.path.exists("buoi09/names.txt"):
    urllib.request.urlretrieve("http://raw.githubusercontent.com/karpathy/makemore/master/names.txt", "buoi09/names.txt")

words = open("buoi09/names.txt", "r").read().splitlines()
print(f"Sô tên: {len(words)}, ví dụ: {words[:5]}")

#stoi / itos
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(f"stoi: {stoi}")
print(f"itos: {itos}")

# Đếm Bigram C (27x27)
N = np.zeros((27,27), dtype=np.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

print("\nBigram counts (5x5 gốc):")
print(N[:5, :5])
print(f"Tổng cặp: {N.sum()}")

# Phan B: chuan hoa P, tinh NLL loss
# Chuẩn hóa counts -> sác xuất
P = N.astype(float)
# Làm mịn +1 để tránh log(0) (smoothing)
P += 1
P /= P.sum(axis=1, keepdims=True)

print("\nP[0, :5] (Từ '.' -> a, b, c, d, e):")
print(P[0, :5])

# Tính NLL loss trên toàn dataset
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_likelihood += np.log(prob)
        n += 1

nll = -log_likelihood / n
print(f"\nNLL loss: {nll:.4f} (Kỳ vọng ~2.45 với smoothing +1)")

# Phan C: sampling sinh ten moi
np.random.seed(0)
print("\n10 tên sinh từ Bigram")
for i in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = np.random.choice(27, p=p)
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))


if __name__ == "__main__":
    pass