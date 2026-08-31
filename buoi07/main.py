# Buoi 7: micrograd - Autograd Engine
# Chay: .\venv\Scripts\python.exe buoi07\main.py
import math

# TODO A: class Value (se viet o day)
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data                # Giá trị thực
        self.grad = 0.0                 # dLoss/dValue
        self._backward = lambda: None   # Hàm tính gradien cục bộ
        self._prev = set(_children)     # node cha
        self._op = _op                  # Phép toán tạo ra node

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad = (1 - t**2) * out.grad

        out._backward = _backward
        return out   

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0   # seed grad Loss  

        for node in reversed(topo):
            node._backward()  

if __name__ == "__main__":
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    d = c + Value(1.0)
    print(f"d.data = {d.data}")
    d.backward()
    print(f"\na.grad = {a.grad} (Kỳ vọng = 3)")
    print(f"b.grad = {b.grad} (Kỳ vọng = 2)")
    print(f"c.grad = {c.grad} (Kỳ vọng = 1)")
    print(f"d.grad = {d.grad} (Kỳ vọng = 1)")

    # Test tanh
    data = 0.5
    x = Value(data)
    t = x.tanh()
    t.backward()
    print(f"\ntanh({data}) = {t.data:.4f}, x.grad = {x.grad:.4f} (kỳ vọng = {1 - t.data**2:.4f})")
