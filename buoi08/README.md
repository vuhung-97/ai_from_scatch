# 🚀 Buổi 8: MLP 2 Tầng Với micrograd — Không Cần Viết Chain Rule Tay
> Nguồn: Karpathy micrograd — xây Neuron / Layer / MLP

---

## 🎯 Mục Tiêu Buổi 8

Dùng **Value** vừa viết ở Buổi 7 để xây **MLP 2 tầng** train trên dataset 4 mẫu Buổi 3-4:

```text
Input 2 -> Hidden 4 (tanh) -> Output 1 (tanh) -> Loss MSE -> backward()
```

Không còn `dLoss/dw` thủ công — chỉ `loss.backward()`!

---

## 1. Kiến Trúc

```text
x = [x1, x2]  (2 chiều)

Hidden: 4 neuron, mỗi neuron:  z = w1*x1 + w2*x2 + b -> tanh(z) -> h_i
Output: 1 neuron:  z = w1*h1 + ... + w4*h4 + b -> tanh(z) -> y_pred

Loss 1 mẫu: (y_true - y_pred)^2
Loss batch: mean(Loss các mẫu)
```

---

## 2. Các Class Cần Viết

- `Neuron(n_in)` : `n_in` weights + bias, `__call__(x)` -> `tanh(sum(w*x)+b)`
- `Layer(n_in, n_out)` : `n_out` Neuron, `__call__(x)` -> list outputs
- `MLP` : `Layer(2,4) -> Layer(4,1)`, `__call__(x)` -> y_pred, `parameters()` -> list Value

---

## 3. Lộ Trình Code (3 phần)

### Phần A — Neuron + Layer
### Phần B — MLP + parameters()
### Phần C — Train loop: forward -> loss -> zero_grad -> backward -> update

---

## 📝 Bài Tập

Train 1000 epochs, learn_rate=0.1 trên data 4 mẫu Buổi 3, in Loss mỗi 200 epochs, đạt Loss <0.01.

---

## ⏭️ Sau Buổi 8

Sang **makemore Bigram** — bắt đầu học ngôn ngữ!
