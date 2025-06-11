import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 配置：请修改以下变量
image_dir = "dclip/print/"  # 图片目录
prompt = "photo of forest in the style of woodcut print."     # 文本提示

# 设备与模型初始化
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 获取图片列表
valid_exts = ('.png', '.jpg', '.jpeg')
images = sorted(
    os.path.join(image_dir, f) for f in os.listdir(image_dir)
    if f.lower().endswith(valid_exts)
)

# 计算文本特征并归一化
text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 对每张图片计算 CLIP 分数
for img_path in images:
    image = Image.open(img_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_features = model.get_image_features(**image_inputs)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    score = (img_features @ text_features.T).item()
    print(f"{os.path.basename(img_path)}: {score:.4f}")
