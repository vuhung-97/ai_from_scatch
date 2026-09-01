# 🚀 Buổi 9: Building makemore — Bigram Language Model (Karpathy Lecture 2)
> Nguồn: [YouTube - Lecture 2 makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) + [repo karpathy/makemore](https://github.com/karpathy/makemore/blob/master/bigram.py) (verbatim)

---

## 🎯 Mục Tiêu Buổi 9 (Mai Học)

Xây **Bigram Language Model** — mô hình ngôn ngữ đơn giản nhất — học từ dataset tên người:

```text
Input: ký tự trước (ví dụ 'a') -> Output: xác suất ký tự tiếp theo (ví dụ 'n' cao)
Ví dụ train trên ["emma", "olivia", ...] -> sinh tên mới "emmma", "olivi..."
```

---

## 1. Tóm Tắt Timestamp Lecture 2 (để đối chiếu video)

| Timestamp | Nội dung |
|-----------|----------|
| 00:00 | Giới thiệu makemore, dataset `names.txt` |
| 08:00 | Bigram counts: đếm cặp ký tự `(a->b)` |
| 22:00 | Chuẩn hóa counts thành xác suất, NLL loss |
| 35:00 | Sampling: sinh tên mới từ Bigram |
| 48:00 | Bigram bằng Neural Net (1 neuron) + gradient |

Bạn **không cần xem video** — đọc tóm tắt + gõ code verbatim dưới đây là đủ. Nếu muốn xem sâu, mở link YouTube đúng timestamp.

---

## 2. Công Thức Cốt Lõi (ASCII)

```text
Bigram counts: C[a, b] = số lần ký tự a đi trước b trong dataset

Xác suất: P(b|a) = C[a, b] / sum_b C[a, b]

Loss NLL 1 từ: -log P(b_đúng | a)
Loss batch: mean(NLL)

Sampling: chọn b tiếp theo theo phân phối P(·|a) (np.random.choice)
```

---

## 3. Lộ Trình Code Buổi 9 (3 phần — mai gõ)

### Phần A — Đọc `names.txt`, xây `stoi`/`itos`, đếm Bigram `C (27×27)`
### Phần B — Chuẩn hóa `P`, tính NLL loss, in loss
### Phần C — Sampling sinh 10 tên mới

---

## 📝 Chuẩn Bị Mai Học

- Không cần tải video
- File `names.txt` sẽ tự tải từ `https://raw.githubusercontent.com/karpathy/makemore/master/names.txt` khi chạy code (hoặc mình sẽ tạo sẵn)
- Chạy CPU 12GB thoải mái (Bigram < 10MB RAM)

---

## ⏭️ Sau Buổi 9

Buổi 10: Bigram bằng Neural Net (thay counts bằng 1 neuron train bằng gradient).
