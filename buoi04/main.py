# Buoi 4: Mini-Project GĐ1 — Tu viet lai mang 3 tang tu nho
# Yeu cau: tu viet sigmoid, Neuron, NeuralNetwork (2 hidden layers) + train()
# Chay: .\venv\Scripts\python.exe buoi04\main.py
import numpy as np

# TODO 1: viet sigmoid va deriv_sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def deriv_sigmoid(output):
    return output * (1 - output)

# TODO 2: viet class Neuron 
class Neuron:
    def __init__(self, weights, bias): 
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

# TODO 3: viet class NeuralNetwork voi kien truc 2 hidden layers
# Vi du: h1,h2 (tang 1) -> h3,h4 (tang 2) -> o1 (output)
class NeuralNetwork:
    def __init__(self):
        np.random.seed(41)
        self.h1 = Neuron(np.random.randn(2), np.random.randn())
        self.h2 = Neuron(np.random.randn(2), np.random.randn())
        self.h3 = Neuron(np.random.randn(2), np.random.randn())
        self.h4 = Neuron(np.random.randn(2), np.random.randn())
        self.o1 = Neuron(np.random.randn(2), np.random.randn())

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_h3 = self.h3.feedforward(np.array([out_h1, out_h2]))
        out_h4 = self.h4.feedforward(np.array([out_h1, out_h2]))

        out_o1 = self.o1.feedforward(np.array([out_h3, out_h4]))
        return (out_h1, out_h2, out_h3, out_h4, out_o1)

    def train(self, data, all_y_trues, learn_rate=0.1, epochs=1000): 
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Tính output của các hàm ẩn
                out_h1, out_h2, out_h3, out_h4, y_pred = self.feedforward(x)
                loss = (y_true - y_pred)**2

                # Backward 
                # o1
                dL_dypred = -2 * (y_true - y_pred)

                dypred_dw9 = deriv_sigmoid(y_pred) * out_h3
                dypred_dw10 = deriv_sigmoid(y_pred) * out_h4
                dypred_db5 = deriv_sigmoid(y_pred)
                dypred_douth3 = deriv_sigmoid(y_pred) * self.o1.weights[0]
                dypred_douth4 = deriv_sigmoid(y_pred) * self.o1.weights[1]

                # h3
                douth3_dw5 = deriv_sigmoid(out_h3) * out_h1
                douth3_dw6 = deriv_sigmoid(out_h3) * out_h2
                douth3_db3 = deriv_sigmoid(out_h3)
                douth3_douth1 = deriv_sigmoid(out_h3) * self.h3.weights[0]
                douth3_douth2 = deriv_sigmoid(out_h3) * self.h3.weights[1]

                # h4
                douth4_dw7 = deriv_sigmoid(out_h4) * out_h1
                douth4_dw8 = deriv_sigmoid(out_h4) * out_h2
                douth4_db4 = deriv_sigmoid(out_h4)
                douth4_douth1 = deriv_sigmoid(out_h4) * self.h4.weights[0]
                douth4_douth2 = deriv_sigmoid(out_h4) * self.h4.weights[1]

                # h1
                douth1_dw1 = deriv_sigmoid(out_h1) * x[0]
                douth1_dw2 = deriv_sigmoid(out_h1) * x[1]
                douth1_db1 = deriv_sigmoid(out_h1)

                # h2
                douth2_dw3 = deriv_sigmoid(out_h2) * x[0]
                douth2_dw4 = deriv_sigmoid(out_h2) * x[1]
                douth2_db2 = deriv_sigmoid(out_h2)

                # Chain Rule
                # h1
                self.h1.weights[0] -= learn_rate * (dL_dypred * dypred_douth3 * douth3_douth1 * douth1_dw1 + dL_dypred * dypred_douth4 * douth4_douth1 * douth1_dw1)
                self.h1.weights[1] -= learn_rate * (dL_dypred * dypred_douth3 * douth3_douth1 * douth1_dw2 + dL_dypred * dypred_douth4 * douth4_douth1 * douth1_dw2)
                self.h1.bias -= learn_rate * (dL_dypred * dypred_douth3 * douth3_douth1 * douth1_db1 + dL_dypred * dypred_douth4 * douth4_douth1 * douth1_db1)

                # h2
                self.h2.weights[0] -= learn_rate * (dL_dypred * dypred_douth3 * douth3_douth2 * douth2_dw3 + dL_dypred * dypred_douth4 * douth4_douth2 * douth2_dw3)
                self.h2.weights[1] -= learn_rate * (dL_dypred * dypred_douth3 * douth3_douth2 * douth2_dw4 + dL_dypred * dypred_douth4 * douth4_douth2 * douth2_dw4)
                self.h2.bias -= learn_rate * (dL_dypred * dypred_douth3 * douth3_douth2 * douth2_db2 + dL_dypred * dypred_douth4 * douth4_douth2 * douth2_db2)

                # h3
                self.h3.weights[0] -= learn_rate * dL_dypred * dypred_douth3 * douth3_dw5
                self.h3.weights[1] -= learn_rate * dL_dypred * dypred_douth3 * douth3_dw6
                self.h3.bias -= learn_rate * dL_dypred * dypred_douth3 * douth3_db3

                # h4
                self.h4.weights[0] -= learn_rate * dL_dypred * dypred_douth4 * douth4_dw7
                self.h4.weights[1] -= learn_rate * dL_dypred * dypred_douth4 * douth4_dw8
                self.h4.bias -= learn_rate * dL_dypred * dypred_douth4 * douth4_db4

                # o1
                self.o1.weights[0] -= learn_rate * dL_dypred * dypred_dw9
                self.o1.weights[1] -= learn_rate * dL_dypred * dypred_dw10
                self.o1.bias -= learn_rate * dL_dypred * dypred_db5

            if epoch % 200 == 0:
                y_preds = np.array([self.feedforward(x)[-1] for x in data])
                mse = ((all_y_trues - y_preds)**2).mean()
                print(f"epoch={epoch}\t->\tmse={mse}")


if __name__ == "__main__":
    # TODO 4: dinh nghia data 4 mau nhu buoi03
    data = np.array([[-0.5,-0.5],[1.0,0.7],[0.2,0.3],[-1.0,-0.8]])
    all_y_trues = np.array([1,0,0,1])

    # TODO 5: tao network, in truoc train, train, in sau train + loss cuoi
    network = NeuralNetwork()
    learn_rate = 0.1
    epochs = 2000
    print(f"\nlearn_rate={learn_rate}, epochs={epochs}\n")
    network.train(data, all_y_trues, learn_rate=learn_rate, epochs=epochs)

    print("\n--- Sau khi train ---")
    for x, y_true in zip(data, all_y_trues):
        y_pred = network.feedforward(x)[-1]
        print(f"x={x} y_true={y_true} -> y_pred={y_pred:.4f} (sai số {abs(y_true-y_pred):.4f})")

    # Tính Loss cuối
    y_preds = np.array([network.feedforward(x)[-1] for x in data])
    loss = ((all_y_trues - y_preds)**2).mean()
    print(f"\nLoss cuối: {loss}")
