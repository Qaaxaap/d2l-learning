import torch
print("torch 版本：", torch.__version__)
print("CUDA 可用：", torch.cuda.is_available())
x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(x, x.shape, x.numel())
if (torch.cuda.is_available()):
    x = x.cuda()
    print(x.device)
    x *= 2
    x = x.cpu()
    print(x)