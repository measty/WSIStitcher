# General purpose WSI stitcher

Intended for WSI level inference using a patch-based model.

It can also be used with model set to None, in which case a dummy 'pass through' model will be created. This can be used for things like format conversion to tiff, masked rebuilding, or mpp conversion.

# Usage:

pip install -r requirements.txt

If using for stitching model outputs, you will have to add your model loading code in the `load_model` function in infer.py.
This will be passed a model checkpoint path, and a config path if provided in the cli command, and is expected to return a PyTorch nn.Module that outputs a torch tensor (torch.float32) of size (B, C, H, W) given an input tensor of size (B, C_in, H, W). The input tensor the model is provided with in the stitching pipeline is just the image patch converted to float in 0-1 via rgb/255 (and with dims shifted to channel-first), so if your model needs any special preprocessing, you will need to add that in a tiny wrapper around your model when you load it.

CLI options provided by `infer.py`:

- `checkpoint`: required path to the trained weights; set to `none` to run the pass-through workflow.
- `--wsi`: path or glob pattern pointing to one or more whole-slide images.
- `--config`: optional YAML configuration for your model (defaults to one colocated with the checkpoint).
- `--tile-size`: tile size in pixels (overrides the training config default).
- `--overlap`: overlap in pixels between adjacent tiles (default `0`).
- `--read-mpp`: microns-per-pixel to sample patches from the input WSI.
- `--save-mpp`: microns-per-pixel to use when writing the output WSI.
- `--batch-size`: number of tiles processed per forward pass (default `16`).
- `--output` / `-o`: output file path or globbed template for batch runs (should be `.tiff` or `.qptiff` if multi-channel (i.e not rgb)).
- `--device`: device identifier such as `cpu`, `cuda`, or `cuda:0` (auto-selects if omitted).
- `--memmap-dir`: optional directory used for intermediate memory-mapped arrays during tiling. defaults to a temp directory in the output folder.
- `--channels`: optional ordered list of channel names for multi-channel inputs.
- `--background-colour`: background colour for compositing the output (`white` default, `black` supported).
- `--mask`: tiatoolbox Tissue mask method (e.g., 'otsu'), path to a mask file, or none for no masking; passed to tiatoolbox SlidingWindowPatchExtractor. Default is 'otsu'.
- `--fliplr`: will save the final slide flipped left-right if set.
- `--crop`: will save the final slide cropped to the masked region if set. (has no effct if --mask is none)

# An example:

a command like:

python infer.py --checkpoint /path/to/checkpoint.pt --wsi "/path/to/slides/*_HE.tiff" --output /path/to/outputs/*_processed.tiff --read_mpp 0.5 --save_mpp 0.25 --overlap 96 --tile-size 512 --batch-size 16

will run inference on all slides in the specified folder matching *_HE.tiff, using the specified checkpoint to load the model. Patches will be read of size tile-size (512) at read_mpp (0.5) (ensure these match the mpp your model is trained at) and will save the output WSI at save_mpp (0.25) mpp (scaling model outputs approriately), with 96 pixels of overlap between tiles for smooth stitching. Outputs will be saved to /path/to/outputs/ with the same stem as the input slides but with _processed.tiff suffix.
