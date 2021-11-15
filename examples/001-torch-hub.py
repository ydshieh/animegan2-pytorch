import torch

from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image


torch.set_grad_enabled(False)
device = "cpu"


model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
model = model.eval().to(device)
print(model)


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
img_path = "./dasha-taran-4-cropped.jpg"
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


# save model
torch.save(model, "model.bin")

