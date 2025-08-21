- calculus /ˈkælkjələs/ n. 微积分
- neuron /ˈnjʊərɒn/ n. 神经元
- in proportion to 与 ... 成比例

- precede 处在 ... 之前
- derivative /dɪˈrɪvətɪv/
  - Partial derivative
  - ![image-20250726211450488](C:\Users\95432\AppData\Roaming\Typora\typora-user-images\image-20250726211450488.png)
  - 导数：derivative
- GPT：Generative Pre-trained Transformer

****

- Questions1

> 在Transformer的编码器（Encoder）中，多头自注意力（Multi-Head Self-Attention）机制的作用是什么？

将输入序列映射到不同的表示子空间，从不同角度捕获信息

- Questions2

> Transformer解码器（Decoder）中的“掩码多头自注意力”（Masked Multi-Head Self-Attention）主要目的是什么？

在序列生成任务中，解码器在预测当前词时只能依赖于已经生成的部分和输入，掩码操作确保了这一点，避免了“偷看”未来的信息。

- Question3

> 在Transformer编码器和解码器中，都包含的前馈网络（Feed-Forward Network）层主要作用是什么？

前馈网络层独立地作用于每个位置，引入了非线性变换，使得模型能够学习更复杂的特征表示。

> 在自注意力（Self-Attention）机制中，计算注意力分数时通常会进行“缩放”（scaling）操作，即除以 $\sqrt {d_k}$。这一操作的主要目的是什么？

防止点积结果过大导致梯度消失或梯度爆炸，稳定训练过程

![image-20250727230312481](C:\Users\95432\AppData\Roaming\Typora\typora-user-images\image-20250727230312481.png)

![image-20250727230443897](C:\Users\95432\AppData\Roaming\Typora\typora-user-images\image-20250727230443897.png)

![image-20250727230838898](C:\Users\95432\AppData\Roaming\Typora\typora-user-images\image-20250727230838898.png)

```cpp
for (int i = 0; i < N; i++)       // 枚举 A 的行
  for (int j = 0; j < K; j++)     // 枚举 B 的列
    for (int k = 0; k < M; k++)   // 枚举中间维度
      C[i][j] += A[i][k] * B[k][j];
```