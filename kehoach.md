# 📖 Kế Hoạch Học Tập AI / Deep Learning From Scratch
> **Dựa trên repo**: [codecrafters-io/build-your-own-x](https://github.com/codecrafters-io/build-your-own-x)  
> **Thư mục dự án**: `D:\CNTT14\hk4\ai_from_scatch`  
> **Môi trường**: Python 3.13 (`.\venv`) • RAM 12GB (chạy CPU, batch_size nhỏ)  
> **Thời gian học**: ~3 giờ/ngày (~21 giờ/tuần)  
> **Ngôn ngữ giảng**: Tiếng Việt 100% (link gốc tiếng Anh giữ nguyên để đối chiếu)  
> **Scope đã khóa**: 20 link (3 `AI Model` + 17 `Neural Network`) — code sâu 7 link Python chính, 10 link Go/JS/C#/F# + 2 Python ứng dụng còn lại chỉ tóm tắt so sánh (cùng logic, khác ngôn ngữ)  
> **Nguyên tắc bám sát repo**: Không tự nghĩ code — mọi code là **verbatim từ repo chính thức** (ví dụ `karpathy/micrograd`, `karpathy/makemore`, `rasbt/LLMs-from-scratch`) kèm link file gốc + timestamp video để bạn đối chiếu. Toán hiển thị dạng ASCII block theo `skill math-formatter`.

---

## 🎯 Mục Tiêu Đạt Được
1. Hiểu sâu bản chất toán học & lập trình đằng sau Neural Networks (không dùng PyTorch/TensorFlow ở giai đoạn đầu).
2. Tự viết được **Autograd Engine**, **Convolutional Neural Network (CNN)** và **GPT (Transformer)** hoàn chỉnh từ đầu.
3. Huấn luyện thành công một phiên bản GPT-mini sinh văn bản.

---

## 📌 Quy Tắc Học Tập
- **No Copy-Paste**: Tất cả code đều phải tự gõ từng dòng để nhớ cú pháp và hiểu luồng dữ liệu.
- **Giải Thích Toán**: Trước khi code bài nào, phải vẽ/viết ra nháp luồng Tensor/Matrix shape.
- **Phá Rồi Sửa**: Kết thúc bài học, thử thay đổi thông số (learning rate, activation, shape) xem lỗi xảy ra thế nào.
- **Bám Sát Repo**: Mỗi `buoi0X/README.md` đầu file đều ghi **link YouTube + link GitHub file verbatim** (ví dụ `micrograd/engine.py#L10-L58`) để bạn kiểm đúng/sai ngay bằng `.\venv\Scripts\python.exe buoi0X/main.py`.

---

## 🗺️ Lộ Trình Chi Tiết Theo Ngày (3h/ngày) — Bám Sát 20 Link Repo

### 🟢 GIAI ĐOẠN 1: Nền Tảng Neural Network thuần NumPy (Ngày 1 - 4) — 2/7 link Python chính
- [x] **Ngày 1**: *A Neural Network in 11 Lines of Python* (iamtrask — Python) — [link](https://iamtrask.github.io/2015/07/12/basic-python-network/)
  - Sản phẩm: `buoi01/main.py` Logic Gate (đã hoàn thành).
- [x] **Ngày 2**: *Implement a Neural Network from Scratch - Phần 1* (Victor Zhou — Python) — [link](https://victorzhou.com/blog/intro-to-neural-networks/)
  - Sản phẩm: `buoi02/main.py` `Neuron` + `NeuralNetwork` feedforward (đã hoàn thành).
- [x] **Ngày 3**: *Implement a Neural Network from Scratch - Phần 2* (Victor Zhou — Python)
  - Sản phẩm: `buoi03/main.py` train 1000 epochs Loss 0.351→0.005 (đã hoàn thành).
- [x] **Ngày 4**: *Mini-Project GĐ1* (tự viết lại 3 tầng `2->2->2->1` không nhìn tài liệu)
  - Sản phẩm: `buoi04/main.py` Loss 0.322→0.0028 (đã hoàn thành).

---

### 🟡 GIAI ĐOẠN 2: Computer Vision & Autograd (Ngày 5 - 17)

#### Phần 2A: Computer Vision cơ bản (Ngày 5 - 6) — 1/7 link Python chính
- [x] **Ngày 5**: *An Introduction to CNNs - Part 1* (Victor Zhou — Python) — [link](https://victorzhou.com/blog/intro-to-cnns-part-1/)
  - Sản phẩm: `buoi05/main.py` `Conv3x3` (đã hoàn thành).
- [x] **Ngày 6**: *An Introduction to CNNs - Part 2* (Victor Zhou — Python)
  - Sản phẩm: `buoi06/main.py` `Conv→MaxPool→Softmax` (đã hoàn thành).

#### Phần 2B: Chuỗi "Neural Networks: Zero to Hero" của Andrej Karpathy (Ngày 7 - 17) — 1/7 link Python chính nhưng 6 lectures
> **Cách học video**: Mình tóm tắt **timestamp + code verbatim từ repo chính thức** (`karpathy/micrograd`, `karpathy/makemore`, `karpathy/nanoGPT`), bạn **không cần tải video** — chỉ đọc tóm tắt + tự gõ code; nếu muốn xem sâu thì mở link YouTube đúng timestamp.

- [x] **Ngày 7 - 8**: *Lecture 1 - Building micrograd* (2h25m) — [YouTube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) + [repo micrograd](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py)
  - Sản phẩm: `buoi07/main.py` `Value` + `buoi08/main.py` MLP 2 tầng micrograd Loss 0.67→0.0009 (đã hoàn thành).
- [x] **Ngày 9 - 10**: *Lecture 2 - Building makemore (Bigram Language Model)* — tóm tắt timestamp + code verbatim từ `karpathy/makemore` (`bigram.py`), NLL loss, Sampling.
  - Sản phẩm: `buoi09/main.py` Bigram counts 228k NLL 2.45 + `buoi10/main.py` Bigram NN W(27,27) train loss 3.3→2.48, sinh `efernviestoricz, f, dedadiaguhi...` (đã hoàn thành).
- [ ] **Ngày 11 - 12**: *Lecture 3 - MLP Language Model (Bengio et al. 2003)* — embeddings, hidden, cross-entropy
- [ ] **Ngày 13 - 14**: *Lecture 4 - Activations, Gradients & BatchNorm* — Kaiming init, vanishing, BatchNorm
- [ ] **Ngày 15 - 16**: *Lecture 5 - Building WaveNet* — dilated convolutions
- [ ] **Ngày 17**: *Lecture 6 - Building GPT from scratch (NanoGPT)* — Self-Attention, Transformer Block

#### Phần 2C: 3 Link Python Chính Còn Lại + Tóm Tắt 10 Link Ngôn Ngữ Khác
- [ ] **Ngày 18**: *Build Deep Learning From Scratch (34 stages)* (Python) — [link](https://github.com/roiamiel1/Build-Deep-Learning-From-Scratch) — so sánh với micrograd
- [ ] **Ngày 19**: *SlowTorch* (Python) — [link](https://github.com/xames3/slowtorch) — PyTorch thuần Python
- [ ] **Ngày 20**: *Traffic signs classification (CNN)* (Python) — ứng dụng CNN thực tế
- [ ] **Ngày 21**: *Tóm tắt 10 link Go/JS/C#/F# + 2 Python ứng dụng còn lại* (`OCR`, `Generate Music LSTM`) — **bảng so sánh 1 trang**: cùng 1 MLP/CNN viết bằng Go/JS khác Python ở syntax nào, không cần code lại (đã đủ 17 link NN).

---

### 🔴 GIAI ĐOẠN 3: AI Model (Ngày 22 - 35) — 3 Link AI Model
> Repo `rasbt/LLMs-from-scratch`, `HuggingFace diffusion`, `langchain-ai/rag-from-scratch` — đều có code verbatim, chạy CPU 12GB với batch_size nhỏ.

- [ ] **Ngày 22 - 24**: *Chapter 2 - Working with Text Data* (Raschka) — Tokenization (BPE), Embeddings
- [ ] **Ngày 25 - 28**: *Chapter 3 - Coding Attention Mechanisms* — Simple/Causal/Multi-Head Attention
- [ ] **Ngày 29 - 32**: *Chapter 4 - Implementing a GPT Model from Scratch* — LayerNorm, GELU, Transformer Block, GPT-2
- [ ] **Ngày 33 - 34**: *Chapter 5 - Pretraining on Unlabeled Data* — Data loader, perplexity, Greedy/Top-k/Temperature
- [ ] **Ngày 35**: *Diffusion Models (HuggingFace)* + *RAG from Scratch* — tóm tắt + demo CPU

> **Ghi chú phần cứng 12GB**: Toàn bộ GĐ3 sẽ chạy `device='cpu'`, `batch_size=1-4`, `num_workers=0` để vừa RAM — chậm hơn GPU nhưng học đủ.

---

## 📊 Bảng Theo Dõi Tiến Độ
- **Tổng số buổi**: 35 buổi (tương đương 5 - 6 tuần với 3h/ngày)
- **Scope**: 20 link (3 AI Model + 17 Neural Network) — code sâu 7 Python chính, còn lại tóm tắt
- **Đã hoàn thành**: 10 / 35 buổi (28.6%) — đã xong Bigram counts + Bigram NN
- **Trạng thái hiện tại**: Hoàn thành Buổi 10 Bigram NN, chuẩn bị Buổi 11-12 — MLP Bengio
