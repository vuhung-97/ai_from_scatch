# 🚀 Buổi 10: Bigram Neural Net — 1 Neuron Thay Bảng Counts (Karpathy Lecture 2 Part 2)
> Nguồn: [YouTube Lecture 2](https://www.youtube.com/watch?v=PaCmpygFfXo) 48:00 + [repo karpathy/makemore — bigram_nn.py](https://github.com/karpathy/makemore)

---

## 🎯 Mục Tiêu Buổi 10

Thay bảng đếm `N (27×27)` của Buổi 9 bằng **1 neuron** train bằng gradient:

```text
Buổi 9:  P = N / sum(N)  (đếm)
Buổi 10: P = softmax(W · one_hot(prev_char))  (học)
```

W shape `(27, 27)` — 27 logits cho mỗi prev_char, train để NLL giảm từ 2.45 xuống ~2.20.

---

## 1. Kiến Trúc

```text
prev_char (one-hot 27) --W(27,27)--> logits 27 --softmax--> probs 27 --NLL--> loss

Ví dụ: prev='.' (index 0) -> one_hot[0]=1 -> logits = W[0] (hàng 0) -> probs
```

---

## 2. Công Thức

```text
logits = W[prev]                    # hàng thứ prev của W
probs  = softmax(logits) = e^logits / sum e^logits
loss   = -log(probs[next])          # NLL cho cặp (prev,next)
```

---

## 3. Lộ Trình Code (3 phần)

### Phần A — Khởi tạo W (27,27) + forward 1 cặp
### Phần B — Train loop 1000 steps, in NLL mỗi 200 steps
### Phần C — Sampling với W đã train

---

## 📝 Bài Tập

So sánh NLL Bigram counts (2.45) vs Bigram NN (~2.20) — NN tốt hơn vì học được smoothing tối ưu, không phải +1 cứng.

---

## ⏭️ Sau Buổi 10

Buổi 11-12: MLP Bengio — embeddings + hidden 100 nơ-ron.
