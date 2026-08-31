# 🚀 Buổi 2: Xây Nơ-ron Đầu Tiên bằng OOP — Bias & Feedforward
> Nguồn: Victor Zhou — *Implement a Neural Network from Scratch (Part 1)*

---

## 🎯 Mục Tiêu Buổi 2

Buổi 1 ta dùng **một ma trận trọng số `W (3,1)`** cho cả 4 mẫu cùng lúc.
Buổi 2 ta **thu nhỏ lại**: hiểu **1 nơ-ron đơn lẻ** hoạt động như thế nào,
rồi ghép nhiều nơ-ron thành mạng — bằng **lập trình hướng đối tượng (OOP)**.

Kết thúc buổi bạn sẽ tự viết được:

```text
Class Neuron  --->  Class NeuralNetwork (2 hidden + 1 output)  --->  feedforward(x) ra dự đoán
```

---

## 1. Nơ-ron Là Gì? So Với Buổi 1

Buổi 1 ta tính:

```text
z = x1*w1 + x2*w2 + x3*w3          (x3 luôn = 1 để đóng vai bias)
y_pred = sigmoid(z)
```

Buổi 2 ta tách rõ vai trò **bias `b`** — một số tự do không nhân với input:

```text
z = w1*x1 + w2*x2 + b
y_pred = sigmoid(z)

Trong đó:
  w1, w2 : trọng số (weights) cho từng đầu vào
  b      : độ lệch (bias) — giúp nơ-ron dịch đường quyết định sang trái/phải
  z      : tổng có trọng số (weighted sum)
  y_pred : đầu ra sau kích hoạt, nằm trong (0, 1)
```

### Tại sao cần bias?

Không có `b`, đường ranh giới của nơ-ron luôn đi qua gốc tọa độ (0,0).
Có `b`, nơ-ron có thể học được ranh giới ở bất kỳ đâu.

```text
Không bias:  z = w1*x1 + w2*x2         --> đường thẳng qua gốc O
Có bias:     z = w1*x1 + w2*x2 + b      --> đường thẳng dịch khỏi gốc O
```

---

## 2. Hàm Sigmoid (nhắc lại, dạng ASCII)

```text
               1
sigmoid(z) = ------------
              1 + e^(-z)

Đạo hàm (dùng ở Buổi 3 khi train):
  sigmoid'(z) = sigmoid(z) * [1 - sigmoid(z)]
```

---

## 3. Kiến Trúc Mạng Buổi 2

Mạng gồm **2 tầng**:

```text
Input (2 chiều)          Hidden (2 nơ-ron)         Output (1 nơ-ron)
                                                        
  x1 ----+----> [ h1 ] --+                                  
         |               +----> [ o1 ] ----> y_pred        
  x2 ----+----> [ h2 ] --+                                  

  h1 = sigmoid(w1*x1 + w2*x2 + b1)
  h2 = sigmoid(w3*x1 + w4*x2 + b2)
  o1 = sigmoid(w5*h1 + w6*h2 + b3)
```

- `h1`, `h2` nhận cùng `x1, x2` nhưng có bộ `w, b` riêng -> học đặc trưng khác nhau.
- `o1` nhận `h1, h2` làm đầu vào.

---

## 4. Dữ Liệu Thử Nghiệm Buổi 2

Ta dùng 4 người, 2 đặc trưng: **cân nặng & chiều cao** (đã chuẩn hóa về gần 0):

```text
   weight_norm = (weight_kg - 60) / 10
   height_norm = (height_cm - 165) / 10

  Alice : [ -0.5 , -0.5 ]  -> y = 1  (nữ)
  Bob   : [  1.0 ,  0.7 ]  -> y = 0  (nam)
  Charlie:[  0.2 ,  0.3 ]  -> y = 0
  Diana : [ -1.0 , -0.8 ]  -> y = 1
```

Mục tiêu hôm nay: **chỉ feedforward** (chưa train), quan sát output ngẫu nhiên.

---

## 5. Lộ Trình Code (3 phần nhỏ)

### Phần A — Class Neuron
Tạo 1 nơ-ron với `weights`, `bias`, phương thức `feedforward(x)`.

### Phần B — Class NeuralNetwork
Ghép `h1, h2, o1` thành mạng, viết `feedforward(x)` toàn mạng.

### Phần C — Chạy thử
Đưa 1 mẫu `x = [-0.5, -0.5]` qua mạng, in `y_pred`.

---

## 📝 Bài Tập
1. Tự gõ Class `Neuron` và `NeuralNetwork` vào `buoi02.py`.
2. Chạy `.\venv\Scripts\python.exe buoi02.py` và quan sát `y_pred`.
3. Trả lời: **Nếu đổi `bias` của `o1` từ 0 thành +5, `y_pred` sẽ tăng hay giảm? Tại sao?**
