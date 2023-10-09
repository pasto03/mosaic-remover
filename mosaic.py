import numpy as np


def add_mosaic(img_2d: np.ndarray, window_size=(9, 9), mode='mean'):
    #     img_2d = np.arange(0, 400).reshape(20, 20)
    assert window_size[0] == window_size[1], "window_size[0] != window_size[1]"
    assert img_2d.ndim == 3, "image dimensions should be [C, H, W]"
    stride = window_size[0]
    C, H, W = img_2d.shape
    new_img_2d = np.zeros_like(img_2d)

    _H, _W = window_size

    for channel_idx in range(C):
        h_i, w_i = 0, 0
        while True:
            h_j = h_i + _H
            w_j = w_i + _W
            img_window = img_2d[channel_idx, h_i:h_j, w_i:w_j]

            if img_window.numel != 0:
                if mode == 'mean':
                    fill = img_window.mean()
                elif mode == 'max':
                    fill = img_window.max()
                elif mode == 'min':
                    fill = img_window.min()

                new_img_2d[channel_idx, h_i:h_j, w_i:w_j] = fill

            w_i += stride

            if h_j >= H and w_j >= W:
                break
            if w_j >= W:
                w_i = 0
                h_i += stride
    return new_img_2d
