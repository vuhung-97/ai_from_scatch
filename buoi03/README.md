# 🚀 Buổi 3: Huấn Luyện Mạng — Loss, Đạo Hàm Riêng & Chain Rule

> Nguồn: Victor Zhou — _Implement a Neural Network from Scratch (Part 2)_ — Train

---

## 🎯 Mục Tiêu Buổi 3

Biến mạng **đoán bừa (0.5~0.6)** ở Buổi 2 thành **đoán chính xác (0.01 / 0.99)** bằng vòng lặp train.

Bạn sẽ tự viết `def train(X, y)` với 4 bước:

```text
1. Feedforward  -> y_pred
2. Tính Loss    -> MSE
3. Tính Gradients bằng Chain Rule (đạo hàm riêng)
4. Cập nhật Weights bằng Gradient Descent
```

---

## 1. Loss Function — Mean Squared Error (MSE)

Với 1 mẫu: `y` là nhãn thật (0 hoặc 1), `y_pred` là dự đoán.

```text
                    1
Loss =  --------------------------------- * sum( (y - y_pred)^2 )
         số_mẫu   (với 1 mẫu thì bỏ 1/số_mẫu)

Ví dụ:
  y = 1, y_pred = 0.6  --> Loss = (1 - 0.6)^2 = 0.16
  y = 0, y_pred = 0.1  --> Loss = (0 - 0.1)^2 = 0.01
```

Loss càng nhỏ --> mạng càng đúng.

---

## 2. Cần Tính Đạo Hàm Để Biết Sửa Weights Thế Nào

Ta cần `dLoss / dWeight` cho từng weight.

Ví dụ với weight `w1` của nơ-ron `h1`:

```text
dLoss/dw1 = dLoss/dy_pred  *  dy_pred/dh1  *  dh1/dw1
            \___________/    \___________/  \_________/
             Loss theo         y_pred theo    h1 theo
             y_pred            h1             w1
```

Đây chính là **Chain Rule (Quy tắc dây chuyền)** — nhân các đạo hàm cục bộ dọc đường đi.

### Công thức đạo hàm cục bộ

```text
a) dLoss/dy_pred  = -2 * (y - y_pred)          (đạo hàm của (y - y_pred)^2)

b) dy_pred/dh1    = w5 * sigmoid'(z_o1)         (y_pred phụ thuộc h1 qua w5)

c) dh1/dw1        = x1 * sigmoid'(z_h1)         (h1 phụ thuộc w1 qua x1)

Trong đó:
              sigmoid'(z) = sigmoid(z) * [1 - sigmoid(z)]
```

Nhân 3 cái lại --> ra `dLoss/dw1`.

---

## 3. Gradient Descent — Cập Nhật Weights

Sau khi có gradient, ta lùi ngược chiều gradient một bước nhỏ `learn_rate`:

```text
w_new = w_old - learn_rate * dLoss/dw

Ví dụ: learn_rate = 0.1
  w1 = 0.5, dLoss/dw1 = 2.0  --> w1_new = 0.5 - 0.1*2.0 = 0.3
```

---

## 4. Kiến Trúc Mạng Buổi 3 (giữ nguyên Buổi 2)

```text
  x1,x2 --> [h1] --+
                   +--> [o1] --> y_pred
  x1,x2 --> [h2] --+
  h1 = sigmoid(w1*x1 + w2*x2 + b1)
  h2 = sigmoid(w3*x1 + w4*x2 + b2)
  o1 = sigmoid(w5*h1 + w6*h2 + b3)
```

Cần tính gradients cho 9 tham số: w1,w2,b1, w3,w4,b2, w5,w6,b3.

---

## 5. Lộ Trình Code Buổi 3

### Phần A — Thêm phương thức `sigmoid_deriv` và lưu `z` trong Neuron

### Phần B — Viết `def train(X_data, y_data, epochs, learn_rate)`

### Phần C — Chạy train 1000 epochs, quan sát Loss giảm và y_pred tiến về 0/1

---

## 📝 Bài Tập Cuối Buổi

1. Train 1000 epochs với learn_rate = 0.1, in Loss mỗi 100 epochs.
2. Thử đổi learn_rate = 1.0 và 0.01 — Loss giảm nhanh hay chậm? Có bị diverge không?
