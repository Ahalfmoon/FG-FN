# FG-FN

# 多模态时间序列插补与预测模型

本项目提出了一个基于Transformer的多模态融合框架,用于医疗时间序列数据的插补和预测任务。该模型可以同时利用结构化的时间序列数据和非结构化的临床文本数据。

## 模型架构

### 1. 多模态特征提取
- 时间序列特征: 使用Transformer编码器处理时序数据
- 文本特征: 使用BioBERT提取临床文本的语义表示
- 变量特征: 为每个时间序列变量生成语义嵌入

### 2. 多模态融合机制
模型包含三个主要的融合模块:

#### 2.1 注意力融合
```python
# 计算文本向量和变量向量之间的注意力得分
scores = torch.matmul(x_vectors, torch.transpose(text_emb, 1, 2))
attn = torch.softmax(scores, dim=-1)
# 基于注意力权重融合文本信息
text_fused = torch.bmm(attn, text_emb)
```

#### 2.2 对比学习
```python
# 计算文本和时间序列表示之间的对比损失
projs = torch.cat((var_flatten, ts_flatten))
contr_logits = projs @ projs.t()
contr_loss = F.cross_entropy(contr_logits, contr_labels)
```

#### 2.3 双线性融合
```python
# 双线性层计算两种模态间的交互
bilinear_fusion = self.bicross256(text_fused256, enc_output)
```

### 3. 联合优化目标
模型同时优化以下目标:
- 时间序列插补损失
- 对比学习损失 
- 预测分类损失

## 主要特点

1. 多模态融合:
- 结合临床文本和时间序列数据
- 利用变量语义信息辅助插补
- 多层次的特征交互机制

2. 端到端训练:
- 插补和预测任务联合优化
- 对比学习促进模态对齐
- 双线性融合增强特征交互

3. 实验验证:
- 在MIMIC-III数据集上进行评估
- 支持多种评估指标(MAE, MSE, RMSE等)
- 可以处理不同缺失率的数据

## 使用方法

1. 数据准备:
```python
# 准备时间序列数据和文本数据
X_intact, X, missing_mask, indicating_mask = mcar(X, 0.05)
X = masked_fill(X, 1 - missing_mask, np.nan)
```

2. 模型训练:
```python
transformer = Transformer(n_steps=48, n_features=59, 
                        n_layers=2, d_model=256)
transformer.fit(X, x_vectors, text_emb, labels)
```

3. 预测与评估:
```python
imputed_data, probs, preds = transformer.impute(X, x_vectors, 
                                               text_emb, labels)
```

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- Transformers
- NumPy
- Scikit-learn

## 引用

如果您使用了本项目的代码,请引用以下论文:
[论文引用信息]
