# Buoi 2: Neuron & Feedforward - khung code, ban tu dien tung phan
# Chay: .\venv\Scripts\python.exe buoi02.py
import numpy as np

# (A) se viet Class Neuron o day
def sigmoid(x):
    """
                   1
    sigmoid = ------------
               1 + e^(-x)
    """
    return 1/(1+np.exp(-x))

# (B) se viet Class NeuralNetwork o day
class Neuron:
    """Một neuron cần 2 thành phần weight, bias"""
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def feedforward(self, input):
        """Tính tổng total = w1.x1 + w2.x2 + b"""
        total = np.dot(input, self.weight) + self.bias

        # Trả ra kết quả đầu ra
        return sigmoid(total)  

class NeuronNetwork:
    def __init__(self):
        np.random.seed(0)

        # Tạo các neruron h1, h2, o1 ngẫu nhiên
        self.h1 = Neuron(np.random.randn(2), np.random.randn()) 
        self.h2 = Neuron(np.random.randn(2), np.random.randn()) 
        self.o1 = Neuron(np.random.randn(2), np.random.randn()) 

    def feedforward(self, x):
        # Hidden network 1: 2 neuron
        h1 = self.h1.feedforward(x)
        h2 = self.h2.feedforward(x)

        # Hidden network 2: 1 neuron
        o1 = self.o1.feedforward(np.array([h1, h2]))

        return o1
# (C) se test o cuoi file

if __name__ == "__main__":
    # Tạo mạng với trọng số ngẫu nhiên
    network = NeuronNetwork()

    # 4 mẫu chuẩn hóa: [weight_norm, height_norm]
    # Phương pháp chuẩn hóa: weight_norm = (weight - 60)/10
    #                        height_norm = (height - 165)/10
    data = {
        "Alice   [-0.5, -0.5]": np.array([-0.5, -0.5]),  # y = 1
        "Bob     [ 1.0,  0.7]": np.array([ 1.0,  0.7]),  # y = 0
        "Charlie [ 0.2,  0.3]": np.array([ 0.2,  0.3]),  # y = 0
        "Diana   [-1.0, -0.8]": np.array([-1.0, -0.8]),  # y = 1
    }

    print("--- Feedforward chưa train ---")
    for name, x in data.items():
        y_pred = network.feedforward(x)
        print(f"{name} -> y_pred = {y_pred:.4f}")

    # Thử nghiệm bias 
    print("\n--- Thử đổi bias của o1 ---")
    print(f"bias o1 ban đầu = {network.o1.bias:.4f}")
    network.o1.bias = 5.0
    print(f"bias o1 sau khi đổi = {network.o1.bias:.4f}")
    for name, x in data.items():
        y_pred = network.feedforward(x)
        print(f"{name} -> y_pred (bias=5.0) = {y_pred:.4f}")