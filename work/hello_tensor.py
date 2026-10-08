import torch
print("torch 版本：", torch.__version__)
print("CUDA 可用：", torch.cuda.is_available())
x = torch.arange(12, dtype=torch.float32)
x.reshape(3, 4)
print(x, x.shape, x.numel())
if (torch.cuda.is_available()):
    x = x.to("cuda");
    print(x.device)
    