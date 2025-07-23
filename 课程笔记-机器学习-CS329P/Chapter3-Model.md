## MLP

Multi-Layer Perceptron，多层感知机。

**本质**：多个线性变换（Linear Layers）叠加 + 非线性激活函数（Nonlinear Activation Functions）

Code (单隐藏层的感知机)

```python
num_inputs = 784
num_hiddens = 256
num_outputs = 10

# 输入X，假设是一个 batch_size = 64的输入
X = torch.randn(64, num_inputs)

def relu(x):
    return torch.clamp_min(x, 0)

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs))

H = relu(X @ W1 + b1)
Y = H @ W2 + b2
```

- `torch.clamp_min(input, min)` 是 PyTorch 中的一个函数，用来对 **张量中每个元素** 进行下限裁剪（clamp），即：如果某个元素小于 `min`，就把它变成 `min`；否则保持不变。

- `nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)`: 从一个标准正态分布（均值为 0，标准差为 1）中随机采样，然后将结果缩小 100 倍

- `A @ B` = `torch.matmul(A, B)`  

## 单通道的卷积

```python
h, w = K.shape
Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
```

## Pooling Layer

二维池化，stride = 1

```python
h, w = K.shape
Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))

for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        if mode == 'max':
            Y[i, j] = X[i:i+h, j:j+w].max()
        elif mode == 'mean':
            Y[i, j] = X[i:i+h, j:j+w].mean()
```

## Simple RNN

$RNN \rightarrow h_{t-1}\rightarrow RNN \rightarrow h_t \rightarrow RNN$ 

```python
W_xh = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
W_hh = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * 0.01)
b_h = nn.Parameter(torch.zeros(num_hiddens))

H = torch.zeros(num_hiddens)
outputs = []

# inputs : (num_steps, batch_size, num_inputs)
for X in inputs:
    H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
    outputs.append(H)
```

- `nn.Parameter` 是 PyTorch 中的一个特殊张量，**它会自动被添加到模型的参数列表中**，并被 `optimizer` 优化。
- `outputs` 保存 RNN 中每个时间步的隐藏状态，供后续预测或分类使用。