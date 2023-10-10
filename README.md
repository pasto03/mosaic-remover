# mosaic-remover
A simple pix2pix approach of mosaic removal


## About model
A trial of __pix2pix__ image translation on mosaic image. Therefore, the model has no difference with normal pix2pix structure.

## Example Usage
Feel free to clone repo to try the pretrained model. Pretrained model saved in _checkpoints_ folder.
<pre>
"""
main.py
"""

...

remover = MosaicRemover(device='cuda')

# 1. apply and remove mosaic
image_path = "ORIGINAL_IMAGE_PATH"
remover(image_path=image_path)

# 2. pass mosaic image path directly -- set apply_mosaic as False
# image_path = "MOSAIC_IMAGE_PATH"
# remover.remove_mosaic(image_path, apply_mosaic=False)
</pre>

## Output
<h3>Example 1</h3>
<div style="display:flex;flex-wrap:wrap">
    <div style="padding:10px;float:left">
    <img src="example outputs/dog8782 -- mosaic.png" alt="mosaic image" width=150px height=150px>
    <p>Mosaic</p>
    </div>
    <div style="padding:10px;float:left">
    <img src="example outputs/dog8782 -- pred.png" alt="mosaic image" width=150px height=150px>
    <p>Prediction</p>
    </div>
    <div style="padding:10px;float:left">
    <img src="example outputs/dog8782 -- original.png" alt="mosaic image" width=150px height=150px>
    <p>Ground truth</p>
    </div>
</div>

<h3>Example 2</h3>
<div style="display:flex;flex-wrap:wrap">
    <div style="padding:10px;float:left">
    <img src="example outputs/dog942 -- mosaic.png" alt="mosaic image" width=150px height=150px>
    <p>Mosaic</p>
    </div>
    <div style="padding:10px;float:left">
    <img src="example outputs/dog942 -- pred.png" alt="mosaic image" width=150px height=150px>
    <p>Prediction</p>
    </div>
    <div style="padding:10px">
    <img src="example outputs/dog942 -- original.png" alt="mosaic image" width=150px height=150px>
    <p>Ground truth</p>
    </div>
</div>

</body>
</html>
## Conclusion
Although pix2pix cannot restore the ground truth perfectly, it gives a possibility that image translation models such as pix2pix can be used in mosaic removal. More approachs will be published to this repo in the future.
