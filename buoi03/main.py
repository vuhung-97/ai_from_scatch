# Buoi 3: Train Neural Network - Victor Zhou Part 2
# Chay tu goc du an: .\venv\Scripts\python.exe buoi03\main.py
# Hoac: cd buoi03; ..\venv\Scripts\python.exe main.py
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(output):
    # output da la sigmoid(z)
    return output * (1 - output)

# --- Se viet Class Neuron va NeuralNetwork o day (copy tu buoi02 va mo rong) ---
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # z = w.x + b -> output = sigmoid(z)
        total = np.dot(self.weights, inputs) + self.bias
        self.last_inputs = inputs
        self.last_total = total
        self.last_output = sigmoid(total)
        return self.last_output


class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.h1 = Neuron(np.random.randn(2), np.random.randn())
        self.h2 = Neuron(np.random.randn(2), np.random.randn())
        self.o1 = Neuron(np.random.randn(2), np.random.randn())

    def feedforward(self, x):
        h1_out = self.h1.feedforward(x)
        h2_out = self.h2.feedforward(x)
        input_hidden_o1 = np.array([h1_out, h2_out])
        o1_out = self.o1.feedforward(input_hidden_o1)
        return o1_out

    def train(self, data, all_y_trues, learn_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Forward ---
                h1_out = self.h1.feedforward(x)
                h2_out = self.h2.feedforward(x)

                input_hidden_o1 = np.array([h1_out, h2_out])
                y_pred = self.o1.feedforward(input_hidden_o1)

                # --- Backward (Chain rule) ---
                # dLoss/dy_pred
                d_L_d_ypred = -2 * (y_true - y_pred)

                # o1 gradients
                d_ypred_d_w5 = h1_out * deriv_sigmoid(y_pred)
                d_ypred_d_w6 = h2_out * deriv_sigmoid(y_pred)
                d_ypred_d_b3 = deriv_sigmoid(y_pred)

                d_ypred_d_h1 = self.o1.weights[0] * deriv_sigmoid(y_pred)
                d_ypred_d_h2 = self.o1.weights[1] * deriv_sigmoid(y_pred)

                # h1 gradients
                d_h1_d_w1 = x[0] * deriv_sigmoid(h1_out)
                d_h1_d_w2 = x[1] * deriv_sigmoid(h1_out)
                d_h1_d_b1 = deriv_sigmoid(h1_out)

                # h2 gradients
                d_h2_d_w3 = x[0] * deriv_sigmoid(h2_out)
                d_h2_d_w4 = x[1] * deriv_sigmoid(h2_out)
                d_h2_d_b2 = deriv_sigmoid(h2_out)

                # chain together
                # o1
                self.o1.weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.o1.weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.o1.bias -= learn_rate * d_L_d_ypred * d_ypred_d_b3
                # h1
                self.h1.weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.h1.weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.h1.bias -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                # h2
                self.h2.weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.h2.weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.h2.bias -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

            # In Loss mỗi 100 epochs
            if epoch % 100 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = ((all_y_trues - y_preds) ** 2).mean()
                print(f"Epoch {epoch:4d} loss: {loss:.4f}")

if __name__ == "__main__":
    # Data: 4 người, 2 đặc trưng đã chuẩn hóa [weight_norm, height_norm]
    # weight_norm = (weight_kg - 60)/10, height_norm = (height_cm - 165)/10
    data = np.array([
        [-0.5, -0.5],  # Alice   -> 1 (nữ)
        [ 1.0,  0.7],  # Bob     -> 0 (nam)
        [ 0.2,  0.3],  # Charlie -> 0
        [-1.0, -0.8],  # Diana   -> 1
    ])
    all_y_trues = np.array([1, 0, 0, 1])

    network = NeuralNetwork()
    print("--- Trước khi train ---")
    for x, y_true in zip(data, all_y_trues):
        y_pred = network.feedforward(x)
        print(f"x={x} y_true={y_true} -> y_pred={y_pred:.4f}")

    # Train
    learn_rate = 0.1
    epochs = 1000
    print(f"\n--- Đang train {epochs} epochs learn_rate={learn_rate} ---")
    network.train(data, all_y_trues, learn_rate=learn_rate, epochs=epochs)

    print("\n--- Sau khi train ---")
    for x, y_true in zip(data, all_y_trues):
        y_pred = network.feedforward(x)
        print(f"x={x} y_true={y_true} -> y_pred={y_pred:.4f} (sai số {abs(y_true-y_pred):.4f})")

    # Tính Loss cuối
    y_preds = np.apply_along_axis(network.feedforward, 1, data)
    loss = ((all_y_trues - y_preds)**2).mean()
    print(f"\nLoss cuối: {loss}")
