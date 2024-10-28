import torch
import gdown
print(torch.cuda.is_available())  # 返回 True 表示支持 CUDA
print(torch.version.cuda)         # 检查 CUDA 版本

