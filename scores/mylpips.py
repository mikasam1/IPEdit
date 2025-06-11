import os
import torch
from PIL import Image
import lpips
import torchvision.transforms as transforms

# 配置：请修改以下变量
reference_image_path = "dclip/ori/forest.jpeg"  # 参考图像路径
image_dir = "dclip/print"  # 待比较图像目录

# 设备初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LPIPS 模型加载
loss_fn = lpips.LPIPS(net='alex').to(device)

# 图像预处理：调整大小、转换为 tensor 并归一化到 [-1,1]
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载并预处理参考图像
ref_img = Image.open(reference_image_path).convert('RGB')
ref_tensor = transform(ref_img).unsqueeze(0).to(device)

# 获取待比较图片列表
valid_exts = ('.png', '.jpg', '.jpeg')
images = sorted(
    os.path.join(image_dir, f) for f in os.listdir(image_dir)
    if f.lower().endswith(valid_exts)
)

# 计算并打印 LPIPS 分数
for img_path in images:
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = loss_fn(img_tensor, ref_tensor)
    print(f"{os.path.basename(img_path)}: {dist.item():.4f}")
