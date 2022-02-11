import json
import os
import sys
import torch

from PIL import Image
from collections import OrderedDict
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator
from model_pt import PTGenerator

sys.path.append("../")


def convert_original_pt_to_pt(pretrained_name, output_name, p="./"):

    local_path = os.path.join(p, f"{pretrained_name}.pt")
    if not os.path.isfile(local_path):
        original_model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained=pretrained_name)
    else:
        original_model = Generator().eval().to(device)
        state_dict = torch.load(local_path, map_location=device)
        original_model.load_state_dict(state_dict)

    original_state_dict = original_model.state_dict()

    model_pt = PTGenerator().eval().to(device)
    state_dict = model_pt.state_dict()

    assert len(original_state_dict) == len(state_dict)
    original_pt_to_pt, pt_to_original_pt = get_conversion_dict(original_state_dict, state_dict, save=True, p=p)

    for name in state_dict:

        weight = state_dict[name]
        original_name = pt_to_original_pt[name]
        original_weight = original_state_dict[original_name]

        # sanity check
        assert weight.size() == original_weight.size()
        assert weight.dtype == original_weight.dtype

        # update `model`'s state_dict
        state_dict[name] = original_weight

    torch.save(state_dict, os.path.join(p, output_name))


def get_conversion_dict(original_state_dict, state_dict, save=False, p="./"):
    """Get the correspondence of model state dictionaries' keys between `bryandlee/animegan2-pytorch` and this
    repository.

    Currently, this only uses the order in the state dictionaries to determine the correspondence.
    This is not generalizable, but it's very simple and works fine with our new PyTorch model implementation.
    """

    original_pt_to_pt = OrderedDict()
    for k1, k2 in zip(original_state_dict, state_dict):
        original_pt_to_pt[k1] = k2
    pt_to_original_pt = {v: k for k, v in original_pt_to_pt.items()}

    if save:
        with open(os.path.join(p, "original_pt_to_pt.json"), "w", encoding="UTF-8") as fp:
            json.dump(original_pt_to_pt, fp, ensure_ascii=False, indent=True)

        with open(os.path.join(p, "pt_to_original_pt.json"), "w", encoding="UTF-8") as fp:
            json.dump(pt_to_original_pt, fp, ensure_ascii=False, indent=True)

    return original_pt_to_pt, pt_to_original_pt


if __name__ == "__main__":

    torch.set_grad_enabled(False)
    device = "cpu"

    # convert the weights
    pretrained_name = "face_paint_512_v2"
    output_name = f"{pretrained_name}_pt.pt"
    p = "./weights/"
    convert_original_pt_to_pt(pretrained_name, output_name=output_name, p=p)

    # create the model and load the weights
    model = PTGenerator()
    model = model.eval().to(device)
    state_dict = torch.load(os.path.join(p, output_name), map_location=device)
    model.load_state_dict(state_dict)

    def face2paint(
        img: Image.Image,
        side_by_side: bool = False,
    ) -> Image.Image:
        img_pt = to_tensor(img).unsqueeze(0) * 2 - 1
        output = model(img_pt.to(device), align_corners=True).cpu()[0]

        if side_by_side:
            output = torch.cat([img_pt[0], output], dim=2)

        output = (output * 0.5 + 0.5).clip(0, 1)

        return to_pil_image(output)

    # Inference
    img_path = "./samples/dasha-taran-4-cropped.jpg"
    width, height = (512, 512)
    channels = 3

    with Image.open(img_path) as img:
        shape = (width, height)
        resized_img = img.resize(shape)
        output_img = face2paint(resized_img, side_by_side=True)
        output_img.show()

    # from torchsummary import summary
    # print(summary(model, input_size=(channels, height, width)))
