# 🚀 Buổi 1: Neural Network Trong 11 Dòng Code (iamtrask)
> Nguồn: `build-your-own-x` → *A Neural Network in 11 lines of Python* + *Sigmoid 11 lines*
> File chính: `buoi01/main.py`

---

## 🎯 Mục Tiêu Buổi 1

Hiểu **luồng học cơ bản nhất** của một mạng nơ-ron chỉ với 1 tầng trọng số:

```text
Input X (4, 3) · Weights W (3, 1)  -->  z (4, 1)  -->  sigmoid(z) --> y_pred (4, 1)
```

Kết thúc buổi bạn tự tay chạy được mạng giải bài toán logic và giải thích được
tại sao `w1 ≈ 9.6` lớn vượt trội.

---

## 1. Dữ Liệu & Quy Luật Ẩn

```text
       Input X                 Target y
  [ Col1, Col2, Col3 ]        [ y ]
  [   0,    0,    1    ]  -->   0
  [   0,    1,    1    ]  -->   0
  [   1,    0,    1    ]  -->   1
  [   1,    1,    1    ]  -->   1

  Col1 = y  (quyết định)
  Col2 = nhiễu (không liên quan)
  Col3 = 1 (bias giả, luôn = 1)
```

Mạng phải tự học: `w1` lớn, `w2` ≈ 0.

---

## 2. Công Thức Cốt Lõi (dạng ASCII)

### Forward pass

```text
z = X · W

               1
y_pred = ------------
          1 + e^(-z)
```

### Loss & Adjustment (Backprop đơn giản)

```text
error       = y - y_pred
deriv       = y_pred * (1 - y_pred)          # sigmoid'(y_pred)
adjustments = error * deriv                  # Hadamard product (từng phần tử)

W_new = W_old + X^T · adjustments
```

Kích thước:

```text
X (4, 3) · W (3, 1)        -> y_pred (4, 1)
X^T (3, 4) · adjustments (4, 1) -> ΔW (3, 1)
```

---

## 3. Các Khái Niệm NumPy Cần Nhớ

| Khái niệm | Ý nghĩa |
|-----------|---------|
| `np.dot(A, B)` | Nhân ma trận thực sự (số cột A == số hàng B) |
| `A * B` (cùng shape) | Hadamard product — nhân từng vị trí |
| `A.T` | Chuyển vị (đổi hàng ↔ cột) |
| `np.exp`, `sigmoid` | Element-wise — áp lên từng phần tử |

---

## 4. Cách Chạy

```powershell
..\venv\Scripts\python.exe main.py
# hoặc từ gốc dự án:
.\venv\Scripts\python.exe buoi01\main.py
```

Kết quả mẫu (seed=1, 10000 epochs):

```text
Weights sau train: [ 9.67, -0.20, -4.62 ]
y_pred:            [0.0096, 0.0078, 0.99, 0.99]  ~ y = [0, 0, 1, 1]
```

---

## 5. Câu Hỏi Ôn Tập

1. `sigmoid_derivative(0.5) = ?` → `0.25` (cực đại, lúc phân vân nhất).
2. `outputs` và `error` là ma trận mà vẫn `sigmoid` được? → Nhờ **element-wise**.
3. `W_new` lớn nhất là `w1` — vì Col1 quyết định y.

---

## 📂 Cấu Trúc Thư Mục Buổi 1

```
buoi01/
  README.md   # tài liệu này
  main.py     # code chính (3 phần: data → sigmoid+weights → training loop)
```

Buổi tiếp theo: `buoi02/README.md` — tách 1 nơ-ron riêng, thêm `bias` tường minh và OOP.
