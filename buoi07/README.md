# 🚀 Buổi 7: micrograd — Tự Viết Autograd Engine (Karpathy Part 1)
> Nguồn: Andrej Karpathy — *Neural Networks: Zero to Hero - Lecture 1: Building micrograd*

---

## 🎯 Mục Tiêu Buổi 7

Tự viết **thư viện Autograd thu nhỏ như PyTorch** — chỉ ~100 dòng — có khả năng:

```text
a = Value(2.0) ; b = Value(3.0)
c = a * b + Value(1.0)  # c = 7.0
c.backward()            # tự tính dC/da, dC/db, ...
```

Đây là **nền tảng để không cần viết Chain Rule tay** như Buổi 3-4 nữa.

---

## 1. Value — Đóng Gói Số + Gradient + Đồ Thị

Mỗi số là 1 node `Value`:

```text
Value {
  data:  giá trị thực (ví dụ 2.0)
  grad:  đạo hàm dLoss/dValue (ban đầu 0)
  _prev: tuple các node cha (để backprop)
  _op:   phép toán tạo ra node ("+", "*", "tanh")
  _backward: hàm tính gradient cục bộ
}
```

---

## 2. Các Phép Toán Cần Cài (Buổi 7)

- `__add__` : c = a + b  ->  dc/da = 1, dc/db = 1
- `__mul__` : c = a * b  ->  dc/da = b, dc/db = a
- `tanh`    : c = tanh(a) -> dc/da = 1 - tanh(a)^2
- `backward()` : duyệt DAG ngược từ Loss về, nhân dồn gradient

---

## 3. DAG Ví Dụ

```text
a=2.0 --+
        +--> c = a*b = 6.0 --+--> d = c + 1 = 7.0  (Loss)
b=3.0 --+                    |
                             +--> d.grad=1 -> c.grad=1 -> a.grad=3, b.grad=2
```

---

## 4. Lộ Trình Code (3 phần)

### Phần A — Class Value (data, grad, _prev, _op)
### Phần B — __add__, __mul__, tanh + _backward
### Phần C — backward() topo sort + test a*b+1

---

## 📝 Bài Tập

1. Tính `a=2, b=-3, c=10, d=a*b + c`, `d.backward()` -> grad của a,b là gì?
2. Thay `*` bằng `+` xem grad đổi thế nào.

---

## ⏭️ Buổi 8

Dùng micrograd xây **MLP 2 tầng** train trên dataset Buổi 3-4 mà không cần viết Chain Rule tay!
