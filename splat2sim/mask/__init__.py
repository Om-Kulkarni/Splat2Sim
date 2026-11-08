"""Robot-mask stage (SAM 3.1 video predictor): frames/ -> masks/.

Mask file convention (``masks/NNNNNN.png``, same stem and size as ``frames/color/NNNNNN.png``):
single-channel uint8, **255 = keep (background), 0 = robot (ignore)**. This is the 3DGRUT / COLMAP /
nerfstudio convention (black = ignore) and the dilated + hole-filled union of all text prompts. 3DGRUT's own
loader wants ``<img>_mask.png`` next to the image (0 = ignore); the splat stage copies/renames these files.
"""
