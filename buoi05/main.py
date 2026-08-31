# Buoi 5: Conv3x3 thuan NumPy
# Chay: .\venv\Scripts\python.exe buoi05\main.py
import numpy as np

# TODO A: ham conv cho 1 vung 3x3 (se viet o day)
def conv_single(region, filter):
    # Tính tổng các tích giữa 2 ma trận -> 1 con số duy nhất
    return np.sum(region * filter)

# TODO B: class Conv3x3 (se viet o day)
class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        # Tạo một filters lưu tất cả filter 3x3, chia 9 để làm nhỏ đầu vào, tránh bùng nổ gradient sau này
        self.filters = np.random.randn(self.num_filters, 3, 3)/9

    def iterater_regions(self, image):
        # image.shape = (h, w)
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                region = image[i:i+3, j:j+3]
                yield region, i, j

    def forward(self, input):
        """Input: (H, W) --> Output: (H-2, W-2, num_filters)"""
        h, w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))
        for region, i, j in self.iterater_regions(input):
            for f in range(self.num_filters):
                output[i, j, f] = conv_single(region, self.filters[f])

        return output


if __name__ == "__main__":
    # Ảnh 5x5 giả: nửa trái = 10, nửa phải = 0 (có cạnh dọc ở cột 2)
    image = np.array([
        [10, 10,  0,  0,  0],
        [10, 10,  0,  0,  0],
        [10, 10,  0,  0,  0],
        [10, 10,  0,  0,  0],
        [10, 10,  0,  0,  0],
    ], dtype=float)
    print("Ảnh 5x5:")
    print(image)

    # Test 1: Conv với 1 filter Sobel dọc (phát hiện cạnh dọc)
    conv = Conv3x3(num_filters=1)
    conv.filters[0] = np.array([[1, 0, -1],
                                [1, 0, -1],
                                [1, 0, -1]], dtype=float)
    output = conv.forward(image)
    print("\nfeature Map 3x3:")
    print(output[:, :, 0])

    # Test 2: 8 filter ngẫu nhiên
    conv8 = Conv3x3(num_filters=8)
    output8 = conv8.forward(image)
    print(f"\n8 filter ngẫu nhiên -> output shape: {output8.shape} (3, 3, 8)")
    print("Feature map filter")
    for i in range(8):
        print(f"\n--- map={i} ---")
        print("- fitler")
        print(conv8.filters[i].round(2))
        print("- output")
        print(output8[:, :, i].round(2))
