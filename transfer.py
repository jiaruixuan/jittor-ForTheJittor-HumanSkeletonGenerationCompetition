import numpy as np

from dataset.asset import Asset
from dataset.format import retarget_mapping, parents

from bone_inspector.extract import extract_asset, ExtractOption

asset = Asset.load("data/train/mixamo/13.npz")
asset.export_fbx("res.fbx")

# # if you need to use prediction results, write code like:
# asset           = Asset.load("data/test/vroid/2011.npz")
# asset.vertices  = np.load("predict/vroid/2011/transformed_vertices.npy")
# asset.parents   = parents
# asset.joints    = np.load("predict/vroid/2011/predict_skeleton.npy")
# asset.skin      = np.load("predict/vroid/2011/predict_skin.npy")
# asset.export_fbx("res.fbx")

tgt = extract_asset("res.fbx",
    ExtractOption(zero_roll=False,extract_mesh=True,extract_track=False,)
)
# load animation
src = extract_asset("data/animation/Backflip.fbx",
    ExtractOption(zero_roll=False,extract_mesh=False,extract_track=True,)
)

# remove bones of toes for better visualization
keep = [v for k, v in retarget_mapping.items() if v != 'l_toe_base' and v != 'r_toe_base']
tgt.keep(keep)

roll = {k: 3.1415926 for (i, k) in enumerate(retarget_mapping) if i >= 6}
src.armature.change_matrix_local(roll=roll)
tgt.armature.change_matrix_local()
tgt.armature = tgt.armature.retarget(src.armature, exact=False, mapping=retarget_mapping)

# export animation
tgt.export_animation("ani.fbx")