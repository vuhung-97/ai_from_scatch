import numpy as np

# Tập dữ liệu đầu vào x (4 mẫu, mỗi mẫu 3 thuộc tính) -> Shape: (4, 3)
X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

# Nhãn mục tiêu là y _> Shape: (4, 1)
# .T là phép chuyển vị (Transpose) để biến mảng hàng 1D thành mảng cột 2D (4, 1)
y = np.array([[0,0,1,1]]).T

print("Shape của X:", X.shape)
print("Shape của y:", y.shape)

# --- Phần 2: HÀM KÍCH HOẠT SIGMOID VÀ TRỌNG SỐ

def sigmoid(x):
    # Biến ọi số x thành giá trị từ 0 đến 1
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    # Tính độ dốc (đạo hàm) khi x đã qua hàm sigmoid
    return x*(1-x)

# Đặt seed cố định để mỗi lần chạy đều ra cùng số ngẫu nhiên
np.random.seed(1)

# Tạo 3 trọng số ngẫu nhiên trong khoảng từ -1 đến 1 (cho 3 đầu vào)
# Ma trận kích thước (3, 1): 3 hàng, 1 cột
weights = 2 * np.random.random((3,1)) - 1

print("\n Trọng số ngẫu nhiên ban đầu:")
print(weights)
print("Shape của weight:", weights.shape)

# --- Phần 3: VÒNG LẶP HUẤN LUYỆN TRAINING LOOP

# chạy 10000 lần để mạng nơ ron học dần
for i in range(10000):
    # Bước 1: Forward Pass (Truyền thẳng)
    input_layer = X
    outputs = sigmoid(np.dot(input_layer, weights))

    # Bước 2: Tính lỗi error
    error = y - outputs

    # Bước 3: Tính lượng điều chỉnh Adjustment
    adjustments = error * sigmoid_derivative(outputs)

    # Bước 4: Cập nhật trọng số update weight
    # np.dot(X.T, adjustments) nhân chuyển vị của X với lượng điều chỉnh
    weights += np.dot(input_layer.T, adjustments)

print("\n----------------")
print("Trọng số SAU KHI huấn luyện (10000 lần)")
print(weights)

print("\n Kết quả dự đoán của Mạng outputs:")
print(outputs)
print("-----------------")

