import numpy as np
import torch
from PIL import Image
from unet import UNet
from torchvision import transforms
import os
import cv2


def predict_img(net, full_img, device):
    net.eval()

    norm_img = transforms.Compose([
        transforms.Resize(256),  # DEBUG: predict 256 images
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    img = norm_img(full_img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        # print(torch.max(output), torch.min(output))
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
    return full_mask


def get_iou(source, thresh=0.3):
    # source = np.array(source)

    source = source / 255

    source = (source > thresh) * 1

    return source*255


def main(fn_image, fn_label_path):
    resize_h = 512
    resize_w = 768
    sub_size = 256
    stride = 256
    overlap = sub_size - stride

    n_row = 2
    n_col = 3

    image = Image.open(fn_image)
    # label = Image.open(fn_label)
    #
    full_w, full_h = image.size

    image_resize = image.resize((resize_w, resize_h))
    image_array = np.array(image_resize)

    prediction = np.zeros((resize_h, resize_w))

    for c in range(n_col):
        for r in range(n_row):
            sub_image = image_array[r*stride:r*stride+sub_size, c*stride:c*stride+sub_size]
            sub_image = Image.fromarray(sub_image.astype(np.uint8))

            mask_soft = predict_img(net=net,
                                    full_img=sub_image,
                                    device=device)

            mask_soft = np.array(mask_soft * 255, dtype=int)
            prediction[r * stride:r * stride + sub_size, c * stride:c * stride + sub_size] = mask_soft


    # prediction = np.array(prediction).astype(np.uint8)
    # otsu_thresh, _ = cv2.threshold(prediction, 0, 255, cv2.THRESH_OTSU)
    # otsu_thresh = otsu_thresh / 255
    # adaptive_thresh = 0.4965 * otsu_thresh + 0.4410
    prediction = get_iou(prediction)
    prediction = np.array(prediction).astype(np.uint8)
    heatmap = cv2.applyColorMap(prediction, cv2.COLORMAP_HOT)
    overlay = cv2.addWeighted(image_array, 0.5, heatmap, 1, 0)
    img_mask_save = fn_label_path.replace('.png', '_overlay.png')
    overlay = Image.fromarray(overlay.astype(np.uint8))
    overlay.save(img_mask_save)

    print('Result has been saved to %s' % img_mask_save)


if __name__ == "__main__":

    net = UNet(n_channels=3, n_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load('./checkpoints/CP_epoch.pth', map_location=device))

    root_dir_image = './_raw_data/val/images/'
    root_dir_label = './_raw_data/val/labels_binary/'
    if not os.path.exists(root_dir_label):
        os.mkdir(root_dir_label)


    imgs = os.listdir(root_dir_image)
    for img in imgs:

        fn_image = os.path.join(root_dir_image, img)
        fn_label = root_dir_label + img

        main(fn_image, fn_label)

