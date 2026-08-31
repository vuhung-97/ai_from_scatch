# 🚀 Buổi 4: Mini-Project GĐ1 — Tự Xây Mạng Sâu Hơn

> Tổng kết Giai đoạn 1 (Buổi 1-3) • Không nhìn tài liệu cũ • Tự thêm 1 hidden layer

---

## 🎯 Mục Tiêu

Bạn đã học:

- Buổi 1: `W (3,1)` + Sigmoid + train 1 tầng
- Buổi 2: `Neuron` + `bias` + `feedforward` OOP
- Buổi 3: `Loss MSE` + ` Chain Rule`+ `Gradient Descent` train 2 tầng (2 hidden + 1 output)

**Buổi 4 bạn tự làm lại từ trí nhớ**, và **nâng cấp mạng lên 3 tầng**:

```text
Gợi ý kiến trúc 3 tầng (tự chọn số nơ-ron, ví dụ 2-2-1):

  Input (2) --> Hidden1 (2 nơ-ron) --> Hidden2 (2 nơ-ron) --> Output (1)

  x1,x2 --> [h1] --+
                   +--> [h3] --+
  x1,x2 --> [h2] --+            +--> [o1] --> y_pred
                                |
  x1,x2 --> [h1] --+            |
                   +--> [h4] --+
  x1,x2 --> [h2] --+

  Cụ thể:
    h1 = sigmoid(w1*x1 + w2*x2 + b1)
    h2 = sigmoid(w3*x1 + w4*x2 + b2)
    h3 = sigmoid(w5*h1 + w6*h2 + b3)
    h4 = sigmoid(w7*h1 + w8*h2 + b4)
    o1 = sigmoid(w9*h3 + w10*h4 + b5)
```

Bạn có thể chọn kiến trúc khác (ví dụ Hidden1=3, Hidden2=2), miễn là **có ít nhất 2 hidden layers**.

---

## 📋 Yêu Cầu Bắt Buộc

1. **Tự viết lại** `sigmoid`, `deriv_sigmoid`, `class Neuron`, `class NeuralNetwork` **không copy** từ `buoi02`/`buoi03`.
2. Mạng có **≥3 tầng** (input không tính), ví dụ `2 -> 2 -> 2 -> 1`.
3. Viết `def train(data, y_trues, learn_rate, epochs)` với **Chain Rule đầy đủ** cho mọi weight/bias mới.
4. Train trên **cùng dataset 4 mẫu** Buổi 3:

```text
  data = [[-0.5,-0.5], [1.0,0.7], [0.2,0.3], [-1.0,-0.8]]
  y    = [ 1, 0, 0, 1 ]
```

5. Đạt **Loss cuối < 0.01** sau ≤2000 epochs với `learn_rate` tự chọn (0.1 hoặc 1.0).

---

## ✅ Tiêu Chí Đạt

Chạy `.\venv\Scripts\python.exe buoi04\main.py` in ra:

```text
Epoch 0    loss: ~0.25-0.35
Epoch 1000 loss: <0.01
...
Sau train: y_pred ≈ [0.9+, 0.05-, 0.05-, 0.9+] sai số <0.1 mỗi mẫu
```

---

## 💡 Gợi Ý Chain Rule Cho 3 Tầng

Với `w1` (thuộc `h1`, tầng 1), gradient phải đi qua **2 tầng sau**:

```text
dLoss/dw1 = dLoss/dy_pred * dy_pred/dh3 * dh3/dh1 * dh1/dw1
                              +
          dLoss/dy_pred * dy_pred/dh4 * dh4/dh1 * dh1/dw1

Vì h1 ảnh hưởng tới cả h3 và h4, nên cộng 2 nhánh lại!
```

Đây là lúc bạn hiểu vì sao **DAG** Buổi 3 quan trọng — với mạng sâu, 1 weight đầu ảnh hưởng nhiều đường.

---

## 📂 Nộp Bài

- Code chính: `buoi04/main.py`
- Khi chạy xong, copy **Loss cuối** và **4 dòng y_pred** gửi để review.
- Nếu Loss không giảm, kiểm tra: (1) có quên `* deriv` ở đâu không, (2) có nhầm `w` nào không, (3) `learn_rate` quá nhỏ/lớn?

---

## ⏭️ Sau Buổi 4

Hoàn thành GĐ1 → sang **GĐ2 Buổi 5: CNN — Conv3x3 thuần NumPy** (lọc ảnh, stride, padding).
