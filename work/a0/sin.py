import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 3))
x = torch.linspace(0, 2 * math.pi, 200)
y = torch.sin(x)
plt.plot(x.numpy(), y.numpy())
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.savefig("work/a0/sin.png", dpi=120)