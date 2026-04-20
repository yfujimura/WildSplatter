from einops import repeat

from depth_anything_3.specs import Gaussians

def repeat_gaussians(gaussians, V_tgt):
    return Gaussians(
        means=repeat(gaussians.means, "b ... -> b v ...", v=V_tgt),
        scales=repeat(gaussians.scales, "b ... -> b v ...", v=V_tgt),
        rotations=repeat(gaussians.rotations, "b ... -> b v ...", v=V_tgt),
        harmonics=repeat(gaussians.harmonics, "b ... -> b v ...", v=V_tgt),
        opacities=repeat(gaussians.opacities, "b ... -> b v ...", v=V_tgt),
    )