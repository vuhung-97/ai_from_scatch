# Buoi 6: MaxPool + Softmax
# Chay: .\venv\Scripts\python.exe buoi06\main.py
import numpy as np

# TODO A: class MaxPool2 (se viet o day)
class MaxPool2:
    def iterate_regions(self, image):
        h, w, num_filters = image.shape
        new_h, new_w = h//2, w//2
        for i in range(new_h):
            for j in range(new_w):
                region = image[i*2:i*2+2, j*2:j*2+2]
                yield region, i, j

    def forward(self, input):
        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters))

        """Tương đương:
        for region, i, j in self.iterate_regions(input):
            for k in range(num_filters):
                output[i, j, k] = np.amax(region[:, :, k])"""

        for region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(region, axis=(0, 1))
        
        return output



# TODO B: class Softmax (se viet o day)
class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes)/9
        self.bias = np.zeros(nodes)

    def forward(self, input):
        input_flat = input.flatten()

        """Công thức hàm softmax
                   exp^(z_i - max(z))
        probs = -----------------------
                 sum_j(exp^(z_j-max(z))
        """
        
        totals = np.dot(input_flat, self.weights) + self.bias
        exp_total = np.exp(totals - np.max(totals))
        probs = exp_total/sum(exp_total)

        return probs

# TODO C: class Conv3x3
class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(self.num_filters, 3, 3)/9

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h-2):
            for j in range(w-2):
                region = image[i:i+3, j:j+3]
                yield region, i, j

    def forward(self, input):
        h, w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))
        for region, i, j in self.iterate_regions(input):
            for k in range(self.num_filters):
                output[i,j,k] = np.sum(region * self.filters[k])

        return output



if __name__ == "__main__":
    # Ảnh 5x5 như Buổi 5
    image = np.array([
        [10, 10,  0,  0,  0],
        [10, 10,  0,  0,  0],
        [10, 10,  0,  0,  0],
        [10, 10,  0,  0,  0],
        [10, 10,  0,  0,  0],
    ], dtype=float)
    image /= 10
    conv = Conv3x3(1)
    conv.filters[0] = np.array([[1, 0, -1],
                              [1, 0, -1],
                              [1, 0, -1]], dtype=float)

    feature = conv.forward(image)

    # pool 2x2 
    pool = MaxPool2()
    pooled = pool.forward(feature)
    print("\n Sau MaxPool2:")
    print(pooled[:, :, 0])
    print(f"shape pooled: {pooled.shape}")

    # softmax input_len=1, nodes=3
    np.random.seed(0)
    softmax = Softmax(input_len=1, nodes=3)
    probs = softmax.forward(pooled)
    print(f"\nProbs: {probs}")
    print(f"Tong probs: {probs.sum():.4f}")
    print(f"Predicted class: {np.argmax(probs)}")
