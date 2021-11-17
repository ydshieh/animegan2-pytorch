import torch

from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

import sys
sys.path.append("../")
from model import Generator
from model_pt import PTGenerator


torch.set_grad_enabled(False)
device = "cpu"

# model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
model = Generator()
model = model.eval().to(device)
model.load_state_dict(torch.load("./weights/face_paint_512_v2.pt", map_location=device))
model_state_dict = model.state_dict()

model_pt = PTGenerator()
model_pt = model_pt.eval().to(device)
model_pt_state_dict = model_pt.state_dict()

print(type(model_state_dict))
print(type(model_pt_state_dict))
print(len(model_state_dict))
print(len(model_pt_state_dict))

# from collections import OrderedDict
# import json
# original_pt_to_pt_correspondence = OrderedDict()
#
# for k1, k2 in zip(model_state_dict, model_pt_state_dict):
#     original_pt_to_pt_correspondence[k1] = k2
#
# pt_to_original_pt_correspondence = {v: k for k, v in original_pt_to_pt_correspondence.items()}
#
# with open("original_pt_to_pt_correspondence.json", "w", encoding="UTF-8") as fp:
#     json.dump(original_pt_to_pt_correspondence, fp, ensure_ascii=False, indent=True)
#
# with open("pt_to_original_pt_correspondence.json", "w", encoding="UTF-8") as fp:
#     json.dump(pt_to_original_pt_correspondence, fp, ensure_ascii=False, indent=True)
#
# exit(0)

import json
with open("pt_to_original_pt_correspondence.json", "r", encoding="UTF-8") as fp:
    pt_to_original_pt_correspondence = json.load(fp)

for k in model_pt_state_dict:

    weight = model_pt_state_dict[k]

    original_k = pt_to_original_pt_correspondence[k]
    original_weight = model_state_dict[original_k]

    assert weight.size() == original_weight.size()
    assert weight.dtype == original_weight.dtype

    model_pt_state_dict[k] = original_weight


torch.save(model_pt_state_dict, "./weights/model_pt_state_dict")


###########################
# Test

model = PTGenerator()
model = model.eval().to(device)
model.load_state_dict(torch.load("./weights/model_pt_state_dict", map_location=device))
model_state_dict = model.state_dict()


def face2paint(
        img: Image.Image,
        side_by_side: bool = False,
) -> Image.Image:
    img_pt = to_tensor(img).unsqueeze(0) * 2 - 1
    output = model(img_pt.to(device)).cpu()[0]

    if side_by_side:
        output = torch.cat([input[0], output], dim=2)

    output = (output * 0.5 + 0.5).clip(0, 1)

    return to_pil_image(output)


# Inference
img_path = "./examples/dasha-taran-4-cropped.jpg"
width, height = (512, 512)
channels = 3

with Image.open(img_path) as img:
    shape = (width, height)
    img_resized = img.resize(shape)

    animed_img = face2paint(img_resized)

    img_resized.show()
    animed_img.show()

from torchsummary import summary

print(summary(model, input_size=(channels, height, width)))
