# PyTorch 面试整理

## Tensor 基础
- Tensor 是 PyTorch 的核心数据结构，类似 NumPy 的 ndarray
- 支持 GPU 加速，自动求导
- 创建: torch.tensor(), torch.zeros(), torch.ones(), torch.randn()
- 属性: .shape, .dtype, .device, .requires_grad

## Autograd 自动求导
原理: 动态计算图 (Dynamic Computation Graph)
- 前向传播时记录操作
- backward() 时自动计算梯度
- requires_grad=True 的张量被追踪
- with torch.no_grad(): 停止追踪（评估/推理时用）
- .detach(): 从计算图中分离

## 训练流程关键代码
```python
optimizer.zero_grad()        # 梯度清零
outputs = model(inputs)      # 前向传播
loss = criterion(outputs, targets)  # 计算损失
loss.backward()              # 反向传播
optimizer.step()             # 更新参数
```

## Dataset 和 DataLoader
- **Dataset**: __len__() 返回大小，__getitem__() 返回样本
- **TensorDataset**: 包装多个 Tensor
- **DataLoader**: batch、shuffle、并行加载
- **collate_fn**: 自定义批次组合逻辑

## nn.Module
- 所有模型的基类
- __init__(): 定义层
- forward(): 前向传播逻辑
- 子模块自动注册，参数自动追踪

## 常见的 Layer
- nn.Linear: 全连接层 Wx + b
- nn.Conv1d/2d: 卷积层
- nn.LSTM/GRU: 循环层
- nn.Embedding: 词嵌入层
- nn.Dropout: 随机丢弃
- nn.BatchNorm1d/2d: 批归一化
- nn.LayerNorm: 层归一化

## 损失函数
- nn.CrossEntropyLoss: 分类（内置 Softmax）
- nn.BCEWithLogitsLoss: 二分类
- nn.MSELoss: 回归
- nn.NLLLoss: 负对数似然

## 模型保存
```python
# 保存全部
torch.save(model.state_dict(), 'model.pt')

# 保存 checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# 加载
model.load_state_dict(torch.load('model.pt'))
```

## 迁移学习
```python
# 冻结预训练层
for param in model.parameters():
    param.requires_grad = False

# 替换分类头
model.classifier = nn.Linear(768, num_classes)
```

## 分布式训练
- DataParallel: 单机多卡，简单但效率低
- DistributedDataParallel: 推荐，多进程，效率高
- FSDP: 全分片数据并行，节省显存

## 调试技巧
- torch.set_grad_enabled(False): 禁用梯度
- torch.autograd.set_detect_anomaly(True): 检测 NaN 梯度
- torch.nn.utils.clip_grad_norm_: 梯度裁剪
- tqdm: 进度条
- TensorBoard: 可视化

## 面试高频手写题
1. 手动实现 Softmax
2. 手动实现 CrossEntropyLoss
3. 手写 Train loop
4. 实现自定义 Dataset
5. 实现简单的 Transformer Block
