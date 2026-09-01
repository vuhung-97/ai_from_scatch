# 📖 Glossary — Từ Điển Thuật Ngữ AI (Buổi 1-10)

> Dùng để tra cứu nhanh khi học theo `build-your-own-x` (20 link). Giảng tiếng Việt, link gốc tiếng Anh giữ nguyên.

| Thuật Ngữ (EN) | Nghĩa Tiếng Việt | Ví Dụ Buổi | Ghi Chú / Công Thức ASCII |
|---|---|---|---|
| **Sigmoid** | Hàm ép số về (0,1) | Buổi 1,3 | `1 / (1 + e^(-z))` |
| **Tanh** | Ép về (-1,1), đạo hàm `1 - t^2` | Buổi 7-8 | `tanh(x) = (e^2x-1)/(e^2x+1)` |
| **Softmax** | Biến logits thành xác suất tổng=1 | Buổi 6,10 | `e^(z_i-max)/sum e^(z_j-max)` |
| **Logits** | Số thô trước Softmax | Buổi 6,10 | `logits = W·x + b` |
| **Weight / Bias** | Trọng số / độ lệch học được | Buổi 1-4,7 | `z = w1*x1 + w2*x2 + b` |
| **Loss / NLL / MSE / Cross-Entropy** | Độ sai: `MSE=(y-yp)^2`, `NLL=-log(p_đúng)` | Buổi 3,9-10 | `Loss = -log(P(b\|a))` |
| **Gradient Descent** | Cập nhật ngược dốc `W -= lr * dL/dW` | Buổi 1,3,8,10 | `lr` = bước chân |
| **Backpropagation / Chain Rule** | Lan truyền gradient `dL/dw = dL/dy * dy/dh * dh/dw` | Buổi 3,7 | Nhân dồn qua DAG |
| **Epoch / Batch / Learning Rate** | 1 epoch=duyệt hết data; batch=nhóm nhỏ (32); lr=bước chân | Buổi 3,8,10 | `lr=0.1` vs `10` |
| **Neuron / Layer / MLP** | 1 nơ-ron `tanh(W·x+b)`; Layer=nhiều Neuron; MLP=chồng Layer | Buổi 2,8,11 | `Layer(2,4) -> Layer(4,1)` |
| **Autograd / DAG / Value** | `Value(data,grad,_prev)` tự tính grad bằng topo sort | Buổi 7-8 | `d.backward()` |
| **Embedding** | Bảng `C(27,10)` biến ký tự -> vector | Buổi 11 | `e = C[ix]` |
| **Bigram** | Mô hình `P(next/prev)` 1 ký tự | Buổi 9-10 | `P = N/sum(N)` vs `softmax(W[prev])` |
| **CNN / Conv3x3 / Filter-Kernel** | Tích chập trượt kernel `sum(region*filter)` | Buổi 5 | `3×3 * 3×3 -> 1 số` |
| **Feature Map / Stride / Padding** | Bản đồ sau Conv; stride=bước trượt; padding=viền 0 | Buổi 5 | `5×5 --3×3--> 3×3` |
| **MaxPool 2×2** | Lấy max vùng `2×2` -> `(H/2,W/2)` | Buổi 6 | `np.amax(region,axis=(0,1))` |
| **Flatten** | Trải `(h,w,F) -> (h*w*F,)` để dot với W | Buổi 6,11 | `input_len = h*w*F` |
| **Sampling** | Chọn ngẫu nhiên theo `probs` để sinh tên | Buổi 9-10 | `np.random.choice(27,p=probs)` |
| **Overfit / BatchNorm / Kaiming** | Sẽ học Buổi 13-14: chuẩn hóa, khởi tạo | Sắp tới | `BatchNorm` giữ mean/var ổn định |

---

**Cách dùng:** Khi quên, `Ctrl+F` tên thuật ngữ trong file này — cột “Ví Dụ Buổi” chỉ tới `buoi0X/README.md` tương ứng.
