---
name: math-formatter
description: Bắt buộc định dạng các công thức toán học và biểu thức toán dưới dạng khung ASCII/Unicode căn chỉnh trực quan khi trao đổi trên giao diện CLI/Terminal. Đảm bảo cực kỳ dễ đọc, tuyệt đối không dùng LaTeX thô ($$...$$).
---

# Skill: Math Formatter cho CLI / Terminal Interface

Skill này bắt buộc AI tuân thủ quy tắc hiển thị các công thức toán học, biểu thức ma trận và chứng minh đại số theo định dạng **văn bản thuần (ASCII block)** thay vì dùng LaTeX thô (`$$ ... $$`).

---

## 🚫 1. BẮT BUỘC KHÔNG DÙNG LaTeX thô

- ❌ **KHÔNG VIẾT**: `$$\sigma(z) = \frac{1}{1 + e^{-z}}$$`
- ❌ **KHÔNG VIẾT**: `$$\Delta w = X^T \cdot (\text{error} \times \text{sigmoid\_derivative}(\hat{y}))$$`
- ❌ **KHÔNG VIẾT**: `$$\frac{d}{dx}f(x)$$`

---

## ✅ 2. LUÔN DÙNG Khung ASCII Block (Code block ```text)

Với các công thức chính, biểu thức toán học hoặc chứng minh từng bước, luôn bọc trong khối ````text ... ```` và căn chỉnh bằng nét đứt/dấu gạch.

### Mẫu 1: Phân số & Hàm số
```text
               1
sigmoid(x) = ------------
              1 + e^(-x)
```

### Mẫu 2: Chứng minh Đạo hàm Sigmoid
```text
           e^(-x)             1          e^(-x)
S'(x) = -------------- = ------------ * ------------
         (1 + e^(-x))^2   1 + e^(-x)     1 + e^(-x)

      = S(x) * [1 - S(x)]
```

### Mẫu 3: Luồng tính toán Neural Network (Forward & Backprop)
```text
1. Forward pass (Truyền xuôi):
   z = X · W
   y_pred = sigmoid(z)

2. Loss / Error (Độ sai lệch):
   error = y - y_pred

3. Adjustments (Độ điều chỉnh):
   delta = error * [y_pred * (1 - y_pred)]

4. Weight Update (Cập nhật trọng số):
   W_new = W_old + (X^T · delta)
```

---

## 📐 3. Quy Tắc Định Dạng Ma Trận & Kích Thước (Shape)

- Luôn ghi rõ kích thước (Shape) ma trận khi nhân:
  ```text
  X (4 hàng, 3 cột) · W (3 hàng, 1 cột)  --->  Result (4 hàng, 1 cột)
  ```
- Dùng ký hiệu Unicode rõ ràng:
  - Phép nhân ma trận: `·` hoặc `np.dot(A, B)`
  - Chuyển vị: `X^T` hoặc `X.T`
  - Đạo hàm: `f'(x)` hoặc `d/dx`
  - Thay đổi / Delta: `Δw` hoặc `adjustments`
