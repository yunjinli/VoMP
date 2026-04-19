from vomp.inference import Vomp
from vomp.inference.utils import save_materials
import sys

input_path = sys.argv[1]

print("Loading VoMP model...")
model = Vomp.from_checkpoint(
    config_path="weights/inference.json",
    use_trt=False 
)

print("Evaluating solid volumetric materials...")
results = model.get_mesh_materials(
    input_path,
    # gpu_device="cuda:0",
    return_original_scale=True,
    # THIS IS THE MAGIC FLAG: 
    # It forces VoMP to output the dense internal volume instead of the hollow surface
    query_points="voxel_centers" 
)

print(f"Extraction complete! Found {results['num_voxels']} solid voxels.")
save_materials(results, "materials.npz")
print("Saved to materials.npz")