
## 安装

建议使用 Python 3.10，并根据显卡环境先安装合适版本的 PyTorch：

```bash
cd UCH-Net
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -e .
```

## 数据组织

RGB 与 X 图像必须具有相同的相对路径和文件名主体，扩展名可以不同：

```text
dataset/
├── images/
│   ├── train/0001.jpg
│   └── val/0002.jpg
├── images_ir/
│   ├── train/0001.png
│   └── val/0002.png
└── labels/
    ├── train/0001.txt
    └── val/0002.txt
```

标签使用标准 YOLO 检测格式：每行 `class x_center y_center width height`，坐标归一化到
0–1。复制 `configs/data.example.yaml` 后修改数据根目录、X 模态目录和类别名称。
X 图像可以是单通道或三通道；单通道会复制为三通道，因此模型输入固定为
`[RGB(3), X(3)]` 六通道。

可以考虑使本项目同时提出的数据集：WRF-Dataset： https://doi.org/10.57760/sciencedb.0148c 

## 训练

```bash
mid-prior-train \
  --data configs/my-data.yaml \
  --scale m \
  --epochs 200 \
  --batch 8 \
  --imgsz 640 \
  --device 0
```

也可以不安装直接运行：

```bash
python train.py --data configs/my-data.yaml --scale m --epochs 200 --device 0
```

`--scale m` 对应原项目示例中的 `scale='m'`。不要再修改 `CUDA_VISIBLE_DEVICES` 的硬编码；
通过 `--device` 选择设备。训练结果默认写入 `runs/train/mid-prior`。


## 快速自检

```bash
pip install -e ".[test]"
pytest -q
```

测试会分别检查六通道训练前向和三通道分析回退。单元测试使用 128×128 方形输入；
矩形批次的短边在深层特征图上会进一步缩小，因此实际训练建议 `imgsz>=256`，正式复现
仍建议使用 32 的倍数（通常为 640）。

## 项目结构

```text
public_release/
├── configs/
│   ├── model.yaml
│   └── data.example.yaml
├── src/mid_prior_yolo/
│   ├── modules.py       # 仅本模型使用的原创/定制模块
│   ├── model.py         # 固定 40 层模型图
│   ├── data.py          # RGB-X 成对数据集
│   ├── trainer.py       # Ultralytics 训练适配器
│   └── cli.py
├── tests/
├── train.py
└── pyproject.toml
```

## 许可

本项目复用并扩展 Ultralytics 组件，因此按 AGPL-3.0-only 发布。
公开或商业使用前请同时审阅 Ultralytics 的许可条款。

