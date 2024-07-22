from skimage.exposure import match_histograms
import numpy as np
import matplotlib.pyplot as plt
import cv2


def _match_cumulative_cdf(source, template, mask = None, mask_source = None):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    Borrowed from: https://github.com/scikit-image/scikit-image/blob/v0.22.0/skimage/exposure/histogram_matching.py#L34-L90
    """
    
    if mask is None:
        print("iden mask")
        mask = np.ones_like(source)

    if mask_source is None:
        mask_source = mask
    
    src_lookup = source[mask_source > 0.5].reshape(-1)
    src_counts = np.bincount(src_lookup, minlength=256)
    tmpl_counts = np.bincount(template[mask > 0.5].reshape(-1), minlength=256)
    tmpl_values = np.linspace(0, 255, 256).astype("uint8")
    # print(src_counts.shape, tmpl_counts.shape)
    # print(tmpl_values.shape)

    
    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source[mask_source > 0.5].size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template[mask > 0.5].size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    # print(interp_a_values[:10], src_lookup[:10])
    # return interp_a_values[src_lookup].reshape(source.shape)


    src_full_lookup = source.reshape(-1)
    # _, src_full_lookup = np.unique(source.reshape(-1),
    #                                                    return_inverse=True,
    #                                                    return_counts=False)                                         


    out = interp_a_values[src_full_lookup]
    # out = np.interp(src_full_lookup, src_lookup, interp_a_values[src_lookup])
    # print(out.min(), out.max(), interp_a_values[src_lookup].min(), interp_a_values[src_lookup].max())
    return out.reshape(source.shape)


def masked_histogram_matching(source, template, mask = None, mask_source = None):

    matched_image = []
    for i in range(source.shape[-1]):

        matched_source = _match_cumulative_cdf(source[..., i], template[..., i], mask, mask_source)
        matched_image.append(matched_source)

    matched_image = np.stack(matched_image, -1)
    # print(matched_image.min(), matched_image.max())
    return matched_image


def resize_image(image, aspect_ratio):

    h, w = image.shape[:2]
    ratio = aspect_ratio[1] / aspect_ratio[0]

    if ratio < 1:
        new_h, new_w = h / ratio, w
    else:
        new_h, new_w = h, ratio * w

    img = cv2.resize(image.copy(), (int(new_w),int(new_h)))

    # input_img = np.array(Image.fromarray(img).resize((w, h), Image.NEAREST))
    return img



def setup_color_correction(image):
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2HSV)
    return correction_target

def apply_color_correction(correction, image):
    # Match hue and saturation

    image_hsv_corrected = match_histograms(
        setup_color_correction(image),
        setup_color_correction(correction),
        channel_axis=-1)
    
    image_in = setup_color_correction(image)
    image_in[..., :2] = image_hsv_corrected[..., :2]
    image = cv2.cvtColor(np.asarray(image_in.copy()), cv2.COLOR_HSV2RGB).astype("uint8")

        


    

    return image


if __name__ == "__main__":

    d_path = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/prompt-to-prompt/ui_outputs/editing/1/"
    im_1 = (plt.imread(d_path + "result_ls.png")[..., :3] * 255).astype("uint8")
    im_m = (plt.imread(d_path + "input_mask.png")[..., 0] * 255).astype("uint8")
    aspect_ratio = np.load(d_path + "image_shape.npy")
    # im_o = (plt.imread(d_path + "output_nt_0_chain.png")[..., :3] * 255).astype("uint8")
    
    print(im_1.max(), im_1.min(), im_1.shape, im_m.shape)
    # print(im_m.max(), im_m.min(), im_m.shape)
    # print(im_o.max(), im_o.min(), im_o.shape)

    out = resize_image(im_1, aspect_ratio).astype("uint8")
    out_m = resize_image(im_m, aspect_ratio).astype("uint8")
    # out = masked_histogram_matching(im_o, im_1, (255 - im_m) / 255.0).astype("uint8")
    # out = masked_histogram_matching(im_o, im_1).astype("uint8")
    plt.imsave(d_path + "result_ls_resized.png", out)
    plt.imsave(d_path + "resized_input_mask.png", out_m, cmap="gray")
    pass