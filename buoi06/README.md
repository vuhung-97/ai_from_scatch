# 🚀 Buổi 6: MaxPool + Softmax — Ghép CNN Hoàn Chỉnh
> Nguồn: Victor Zhou — *CNNs Part 2* (MaxPool, Softmax, MNIST)

---

## 🎯 Mục Tiêu Buổi 6

Nối tiếp `Conv3x3` (Buổi 5), thêm 2 lớp còn lại để có **CNN mini** chạy được:

```text
Input (28×28) --> Conv3x3 (8 filters) --> MaxPool 2×2 --> Softmax (10 classes) --> y_pred
```

---

## 1. MaxPool 2×2 — Giảm Kích Thước, Giữ Đặc Trưng Mạnh Nhất

Lấy **max** trong mỗi vùng `2×2`, stride `2` (không chồng):

```text
Vùng 2×2:          MaxPool:
[[3, 1],   -->     [[3]]
 [2, 0]]           (lấy max = 3)

Feature Map 4×4:        Pool 2×2:
[[1, 3, 2, 0],          [[3, 4],
 [2, 0, 1, 4],   -->     [2, 4]]
 [0, 1, 3, 2],
 [1, 2, 0, 4]]
```

Kích thước:

```text
Input (H, W, F) --MaxPool 2×2--> (H/2, W/2, F)

Ví dụ: (26, 26, 8) --Conv--> (13, 13, 8) sau Pool
       (4, 4, 8) --> (2, 2, 8)
```

---

## 2. Softmax — Biến Logits Thành Xác Suất

Đầu ra CNN là 10 số `logits` (chưa chuẩn hóa) cho 10 chữ số 0-9.
Softmax biến thành xác suất tổng = 1:

```text
                 e^(z_i)
softmax(z_i) = ------------
                sum_j e^(z_j)

Ví dụ: logits = [2.0, 1.0, 0.1]  -->  softmax ≈ [0.65, 0.24, 0.09]  (tổng = 1.0)
       -> Dự đoán class 0 (xác suất 65%)
```

---

## 3. Cross-Entropy Loss

```text
Loss = -log( p_đúng )

Ví dụ: nhãn đúng = class 0, p0 = 0.65  --> Loss = -log(0.65) = 0.43
       nếu p0 = 0.99  --> Loss = 0.01 (rất nhỏ)
       nếu p0 = 0.05  --> Loss = 2.99 (rất lớn)
```

---

## 4. Lộ Trình Code (3 phần)

### Phần A — Class `MaxPool2`
- `iterate_regions` duyệt vùng 2×2 stride 2
- `forward(input)` lấy max mỗi vùng

### Phần B — Class `Softmax`
- `forward(input)` flatten (H,W,F) -> vector, nhân weights, cộng bias, softmax
- Khởi tạo `weights` shape (H*W*F, 10) và `bias` shape (10,)

### Phần C — Ghép Pipeline
```
image 5×5 -> Conv3x3(1 filter) -> MaxPool -> Softmax(3 classes) -> probs
```
Test trên ảnh 5×5 Buổi 5, in `probs` và `predicted class`.

---

## 📝 Bài Tập

1. Thử MaxPool trên Feature Map 3×3 Sobel Buổi 5 — kết quả 1×1 là gì?
2. Thử Softmax với logits `[5, 1, 1]` và `[1, 1, 1]` — xác suất thay đổi thế nào?
3. (Nâng cao) Train 1 epoch trên MNIST thật (sẽ làm ở buổi sau nếu muốn).

---

## ⏭️ Sau Buổi 6

Hoàn thành **GĐ2A CNN** → sang **GĐ2B Karpathy micrograd** (tự viết Autograd DAG)!
