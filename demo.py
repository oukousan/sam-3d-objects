# Copyright (c) Meta Platforms, Inc. and affiliates.
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

# import inference code
sys.path.append(str(ROOT / "notebook"))
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = ROOT / "checkpoints" / tag / "pipeline.yaml"
inference = Inference(str(config_path), compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image_dir = ROOT / "notebook" / "images" / "shutterstock_stylish_kidsroom_1640806567"
image = load_image(image_dir / "image.png")
mask = load_single_mask(image_dir, index=14)

# run model
output = inference(
    image,
    mask,
    seed=42,
    with_mesh_postprocess=False,
    with_texture_baking=False,
    use_vertex_color=False,
)

# export gaussian splat
splat_path = ROOT / "splat.ply"
output["gs"].save_ply(splat_path)
print(f"Your Gaussian splat has been saved to {splat_path}")

# export textured mesh. GLB embeds the baked texture in the file.
mesh_path = ROOT / "textured_mesh.glb"
if output["glb"] is None:
    raise RuntimeError("The pipeline did not return a mesh/GLB output.")
output["glb"].export(mesh_path)
print(f"Your textured mesh has been saved to {mesh_path}")
