# ML Forge 项目说明

## 项目是做什么的

ML Forge 是一个基于 Dear PyGui 构建的可视化机器学习桌面应用，目标是让用户在不直接编写训练代码的情况下，通过拖拽节点、连接流程图的方式，搭建并训练 PyTorch 图像模型。

项目当前聚焦在图像任务，尤其是图像分类场景。用户围绕 `Data Prep`、`Model`、`Training` 三个标签页完成数据准备、模型搭建和训练配置，程序会据此：

- 校验整条训练流水线是否完整
- 在后台构建 PyTorch 模型、数据集和 DataLoader
- 执行训练并实时回传日志、损失和精度
- 保存 checkpoint
- 导出独立的 PyTorch 训练脚本
- 使用训练后的权重做推理

## 具体有哪些功能

### 1. 可视化数据准备

- 支持通过节点方式配置数据链路，而不是手写 `torchvision` 代码
- 内置数据集节点：
  - `MNIST`
  - `FashionMNIST`
  - `CIFAR10`
  - `CIFAR100`
  - `ImageFolder`
- 内置数据增强/预处理节点：
  - `Resize`
  - `CenterCrop`
  - `RandomCrop`
  - `RandomHFlip`
  - `RandomVFlip`
  - `ColorJitter`
  - `RandomRotation`
  - `GaussianBlur`
  - `RandomErasing`
  - `Normalize`
  - `ToTensor`
  - `Grayscale`
- 支持训练集和验证集两种模式：
  - 单链路 `DataLoader (train)`，再按 `val_split` 自动切分验证集
  - 双链路分别接 `DataLoader (train)` 和 `DataLoader (val)`，独立定义训练/验证数据流

### 2. 可视化模型搭建

- 通过节点搭建模型结构，包含输入、输出、层、激活函数、归一化和池化模块
- 当前内置的模型节点包括：
  - 层：`Linear`、`Conv2D`、`ConvTranspose2D`、`Flatten`
  - 激活：`ReLU`、`Sigmoid`、`Tanh`、`GELU`、`Softmax`、`LeakyReLU`
  - 归一化/正则：`BatchNorm2D`、`LayerNorm`、`GroupNorm`、`Dropout`
  - 池化：`MaxPool2D`、`AvgPool2D`、`AdaptiveAvgPool2D`
  - I/O：`Input`、`Output`
- 支持根据连线做自动补全和推断：
  - 根据数据集自动填充 `Input` 形状和分类数
  - 自动传播 `in_channels`、`out_channels`、`in_features`、`out_features`
  - `Flatten` 后自动计算下一个 `Linear` 的 `in_features`
  - 对维度不匹配的节点做高亮提示

### 3. 可视化训练编排

- 训练页采用独立的训练图，把数据、模型、损失函数、优化器显式串起来
- 自动提供并锁定 `ModelBlock` 和 `DataLoaderBlock`
- 支持的损失函数：
  - `CrossEntropyLoss`
  - `MSELoss`
  - `BCELoss`
  - `BCEWithLogits`
  - `NLLLoss`
  - `HuberLoss`
  - `KLDivLoss`
- 支持的优化器：
  - `Adam`
  - `AdamW`
  - `SGD`
  - `RMSprop`
  - `Adagrad`
  - `LBFGS`
- 训练前会做整体验证，检查：
  - 三个角色页签是否存在
  - 数据链路、模型链路、训练链路是否完整
  - 必要输入输出是否接通
  - 参数是否为空
  - 图中是否存在环

### 4. 训练控制与实时反馈

- 提供 `RUN`、`PAUSE`、`STOP` 控制按钮
- 后台线程执行训练，避免阻塞界面
- 支持设备选择：
  - `auto`
  - `cuda`
  - `cpu`
  - `mps`
- 支持 AMP 混合精度训练
- 菜单栏实时显示 CUDA 可用性和显存占用
- 训练过程中实时更新：
  - 控制台日志
  - 进度条
  - batch loss
  - train loss
  - val loss
  - val accuracy
- 训练中的 `ModelBlock` / `DataLoaderBlock` 标题会动态显示 epoch 和指标

### 5. Checkpoint 与早停

- 支持配置 checkpoint 保存目录
- 支持按周期保存 checkpoint
- 支持只保存最佳模型
- 可按 `val_loss`、`val_acc` 或 `train_loss` 监控最佳结果
- 训练结束会额外保存 `final.pth`
- 支持 Early Stopping：
  - 开关
  - `patience`
  - `min_delta`

### 6. 指标分析与模型摘要

- 训练后可打开 Metrics 窗口查看最近一次运行结果
- 指标窗口展示：
  - 最终 train loss / val loss / val accuracy
  - 最佳 epoch
  - 拟合诊断信息
  - loss 曲线
  - accuracy 曲线
  - batch loss 曲线
- 右侧面板内置模型摘要区，可估算：
  - 每层参数量
  - 总参数量
  - 估算显存占用（fp32）

### 7. 推理与导出

- 支持从已保存的 `.pth` checkpoint 加载模型并执行推理
- 推理流程会从 Data Prep 图中恢复测试/验证数据集，随机抽样图片后展示：
  - 样本预览
  - 真实标签
  - Top-k 预测结果
  - 每个类别的置信度
- 支持将当前可视化流程导出为独立的 PyTorch `train.py`
- 导出的脚本包含：
  - 数据集与 transforms 构建
  - 模型定义
  - loss / optimizer 初始化
  - 训练与验证循环

### 8. 工程与交互能力

- 支持项目保存与加载，文件格式为 `.mlf` / JSON
- 内置模板：
  - `MNIST Classifier`
  - `CIFAR10 Classifier`
- 支持多标签画布和标签角色分配
- 支持撤销/重做
- 支持删除节点、清空画布
- 支持搜索节点面板
- 支持状态栏、控制台日志、帮助文档和 About 窗口
- 内置快捷键：
  - `Ctrl+S`
  - `Ctrl+Z`
  - `Ctrl+Y`
  - `Del`
  - `Ctrl+Backspace`

## 项目整体流程

典型使用路径如下：

1. 在 `Data Prep` 页签配置数据集、预处理和 DataLoader
2. 在 `Model` 页签用节点搭建神经网络
3. 在 `Training` 页签连接 `DataLoaderBlock -> ModelBlock -> Loss -> Optimizer`
4. 配置 epochs、device、checkpoint、early stopping
5. 点击 `RUN` 开始训练
6. 训练结束后查看 Metrics、加载 checkpoint 做推理，或导出 PyTorch 代码

## 主要代码结构

- `ml_forge/main.py`
  - 应用入口，负责初始化窗口、菜单、主布局和主循环
- `ml_forge/ui/`
  - 界面层，包含菜单、布局、训练面板、控制台、状态栏、摘要等
- `ml_forge/graph/`
  - 节点画布、标签页、连线、撤销重做等交互逻辑
- `ml_forge/engine/blocks.py`
  - 所有节点定义，是理解支持功能范围的核心文件
- `ml_forge/engine/graph.py`
  - 图构建、拓扑排序、整体验证逻辑
- `ml_forge/engine/autofill.py`
  - 自动补全、维度传播、形状推断
- `ml_forge/engine/run.py`
  - 训练线程、模型构建、数据集/DataLoader 构建、checkpoint 保存
- `ml_forge/engine/generator.py`
  - 将当前流程导出为可运行的 PyTorch 脚本
- `ml_forge/engine/inference.py`
  - checkpoint 推理与样本展示
- `ml_forge/filesystem/save.py`
  - 项目保存/加载

## 当前项目定位

这个项目本质上是一个面向初学者或希望快速试验模型结构的可视化 PyTorch 训练器。它强调：

- 低代码或无代码搭建训练流程
- 桌面 GUI 交互
- 图像分类任务的快速实验
- 训练、推理、指标分析、代码导出的闭环

如果后续继续扩展，最值得优先关注的入口通常是：

- `ml_forge/engine/blocks.py`：新增节点类型
- `ml_forge/engine/run.py`：新增训练时真实执行能力
- `ml_forge/engine/generator.py`：新增导出代码映射
- `ml_forge/engine/graph.py`：补充验证规则
