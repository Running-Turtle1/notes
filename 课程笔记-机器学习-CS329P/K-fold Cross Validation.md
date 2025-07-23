## K折交叉验证

K折交叉验证（K-Fold Cross Validation）是一种常用的模型评估方法，特别适用于样本数量不大时评估模型的泛化能力。它的主要思想是将数据集划分为 K 个大小相近的子集（称为“折”），然后进行 K 次训练与验证，使每个子集都恰好被用作一次验证集，其余 K-1 个子集作为训练集。


### 工作流程如下：


假设我们进行 **K=5 折交叉验证**，流程是：


1. 将数据随机划分为 5 个子集（Fold1, Fold2, ..., Fold5）。
2. 第一次：用 Fold1 做验证集，其余 Fold2~Fold5 做训练集。
3. 第二次：用 Fold2 做验证集，其余 Fold1, Fold3~Fold5 做训练集。
4. ...
5. 第五次：用 Fold5 做验证集，其余 Fold1~Fold4 做训练集。
6. 对每一次的模型在验证集上的性能打分。
7. 最终计算这 K 次验证结果的 **平均性能指标（如准确率、F1分数、MSE 等）**，作为模型在该数据集上的整体评估。


### 优点：


- **更加稳定可靠**：避免单一训练/测试划分带来的偶然性。
- **充分利用数据**：每个样本都被用作训练和验证，提高了数据利用率。
- **适用于小数据集**：特别适合样本较少的场景，能够提升模型评估的稳定性。


### Python 中的使用示例（以 scikit-learn 为例）：


```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

X = ...  # 特征
y = ...  # 标签

kf = KFold(n_splits=5, shuffle=True, random_state=42) # kf 函数
model = LogisticRegression()

scores = []

for train_index, val_index in kf.split(X): // split 是 kf 的函数, 返回一个可迭代对象
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    scores.append(accuracy_score(y_val, y_pred))

print("平均准确率：", np.mean(scores))
```

输出 :

```yaml
Train: [2 3 4] Validation: [0 1]
Train: [0 1 4] Validation: [2 3]
Train: [0 1 2 3] Validation: [4]
```

- `KFold` 默认不会打乱数据（`shuffle=False`），如果想要随机划分，请设置 `shuffle=True` 并设定 `random_state` 保证复现。

- `KFold.split()` 只是负责划分索引，**不训练模型**，你需要在循环里自己处理模型训练与评估。

