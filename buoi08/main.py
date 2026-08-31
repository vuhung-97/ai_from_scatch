# Buoi 8: MLP 2 tang voi micrograd
# Chay: .\venv\Scripts\python.exe buoi08\main.py
import random
import math
# Copy class Value tu buoi07/main.py vao day (hoac import)
# from buoi07.main import Value  # se tu viet lai de nho

# TODO A: class Value
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._children = _children
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def backward(self):
        topo = []
        visitted = set()
        def build_topo(v):
            if v not in visitted:
                visitted.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
        

# TODO B: class Neuron, Layer
class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # x: list Value length n_in
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        

class MLP:
    def __init__(self, n_in, n_hidden, n_out):
        self.l1 = Layer(n_in, n_hidden)
        self.l2 = Layer(n_hidden, n_out)

    def __call__(self, x):
        # x: list Value
        h = self.l1(x)
        # h là list 4 Value, l2 cần list Value -> ok
        out = self.l2(h)
        return out

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


if __name__ == "__main__":
    # Data 4 mẫu như Buổi 3 (đã chuẩn hóa)
    data = [[-0.5, -0.5], [1.0, 0.7], [0.2, 0.3], [-1.0, -0.8]]
    y_trues = [1, 0, 0, 1]

    mlp = MLP(n_in=2, n_hidden=4, n_out=1)

    epochs = 1000
    for epoch in range(epochs):
        # Forward + Loss
        y_preds = [mlp([Value(x[0]), Value(x[1])]) for x in data]
        # MSE loss: mean((y_pred - y_true)**2)
        loss = sum(((yp - Value(yt)) * (yp - Value(yt)) for yp, yt in zip(y_preds, y_trues)), Value(0)) * Value(1.0/len(y_trues))

        # Backward
        mlp.zero_grad()
        loss.backward()

        # Update (SGD)
        learn_rate = 0.1
        for p in mlp.parameters():
            p.data -= learn_rate * p.grad

        if epoch % 200 == 0:
            print(f"Epoch={epoch:4d} loss {loss.data:.4f}")

    print("\nSau train:")
    for x,yt in zip(data, y_trues):
        yp = mlp([Value(x[0]), Value(x[1])])
        print(f"x={x} y_true={yt} -> y_pred={yp.data:.4f}")
