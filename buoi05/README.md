# 🚀 Buổi 5: Conv3x3 Thuần NumPy — Convolution Filter
> Nguồn: Victor Zhou — *CNNs Part 1* + build-your-own-x CNN

---

## 🎯 Mục Tiêu Buổi 5

Hiểu **phép tích chập (Convolution)** — viên gạch nền của mọi CNN —
và tự viết `class Conv3x3` chỉ với NumPy (không dùng PyTorch).

Kết thúc buổi bạn sẽ:

```text
Input ảnh (H, W)  --Conv3x3-->  Feature Map (H-2, W-2)  --ReLU-->  ...
```

---

## 1. Convolution Là Gì?

Lấy một **kernel 3×3** trượt khắp ảnh, mỗi vị trí tính **tổng tích từng phần tử**:

```text
Output[i, j] = sum(m=0..2) sum(n=0..2)  Image[i+m, j+n] * Filter[m, n]

Ví dụ kernel phát hiện cạnh dọc:
  Filter = [[ 1, 0, -1],
            [ 1, 0, -1],
            [ 1, 0, -1]]

Vùng ảnh 3x3:      Nhân từng cặp rồi cộng:
  [[10, 10,  0],      10*1 + 10*0 + 0*(-1) = 10
   [10, 10,  0],      10*1 + 10*0 + 0*(-1) = 10
   [10, 10,  0]]      10*1 + 10*0 + 0*(-1) = 10  => Output = 30 (cạnh mạnh!)
```

---

## 2. Kích Thước Output

Với ảnh `H × W`, kernel `3×3`, stride `1`, không padding:

```text
H_out = H - 3 + 1 = H - 2
W_out = W - 3 + 1 = W - 2

Ví dụ: ảnh 5×5 --Conv3x3--> 3×3
       ảnh 28×28 (MNIST) --> 26×26
```

Nếu dùng **padding = 1** (viền 0 quanh ảnh) thì giữ nguyên kích thước:

```text
H_out = H , W_out = W   (vì (H+2) -2 = H)
```

Buổi 5 ta làm **không padding** cho đơn giản.

---

## 3. Nhiều Filter = Nhiều Feature Map

1 filter → 1 feature map (phát hiện 1 loại đặc trưng, ví dụ cạnh dọc).
`num_filters = 8` → 8 feature maps, mỗi filter học 1 mẫu khác nhau.

```text
Input (H, W)  --8 filters 3x3-->  Output (H-2, W-2, 8)
```

---

## 4. Lộ Trình Code (3 phần)

### Phần A — Hàm `conv_single` (1 filter, 1 vùng)
Viết hàm trượt kernel 3×3 trên 1 vùng ảnh 3×3, trả về 1 số.

### Phần B — Class `Conv3x3`
- `__init__(num_filters)` khởi tạo `self.filters` shape `(num_filters, 3, 3)` ngẫu nhiên /9
- `forward(input)` trượt khắp ảnh, tạo output `(H-2, W-2, num_filters)`
- Thêm `iterate_regions` helper để duyệt từng vùng 3×3

### Phần C — Chạy thử trên ảnh 5×5 giả
Tạo ảnh 5×5 đơn giản, dùng 1 filter Sobel, in Feature Map 3×3.

---

## 📝 Bài Tập Cuối Buổi

1. Thử 2 filter khác nhau: Sobel dọc và Sobel ngang, so sánh 2 feature map.
2. Thử `num_filters=8` ngẫu nhiên — quan sát 8 feature maps khác nhau.
3. (Nâng cao) Thêm padding=1 để output giữ nguyên 5×5.

---

## ⏭️ Buổi 6

`MaxPool 2×2` + `Softmax` + `Cross-Entropy` → ghép Conv + Pool + Softmax thành CNN hoàn chỉnh nhận diện MNIST.
