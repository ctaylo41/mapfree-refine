import numpy as np, torch
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images
import cv2


class MAST3RWrapper():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        self.model = AsymmetricMASt3R.from_pretrained(self.model_name).to(self.device)

    def infer(self, img0_path, img1_path, K0, K1):
        img0_size = cv2.imread(img0_path).shape
        img1_size = cv2.imread(img1_path).shape

        images = load_images([img0_path, img1_path], size=512)

        # true_shape is [H, W]; [-2:] -> [H, W], so [0]=H, [1]=W
        img0_resize_size = np.array()(images[0]["true_shape"]).reshape(-1)[-2:]
        img1_resize_size = np.array(images[1]["true_shape"]).reshape(-1)[-2:]

        # sx = resized_W / orig_W ; sy = resized_H / orig_H
        img0_sx, img0_sy = img0_resize_size[1]/img0_size[1], img0_resize_size[0]/img0_size[0]
        img1_sx, img1_sy = img1_resize_size[1]/img1_size[1], img1_resize_size[0]/img1_size[0]

        K0_resize = K0.copy()
        K1_resize = K1.copy()

        K0_resize[0][0] *= img0_sx   # fx
        K0_resize[0][2] *= img0_sx   # 'x
        K0_resize[1][1] *= img0_sy   # fy
        K0_resize[1][2] *= img0_sy   # cy

        K1_resize[0][0] *= img1_sx
        K1_resize[0][0] *= img1_sx
        K1_resize[1][1] *= img1_sy
        K1_resize[1][2] *= img1_sy

        with torch.no_grad():
            output = inference([tuple(images)], self.model, self.device, batch_size=1, verbose=False)

        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        desc1 = pred1['desc'].squeeze(0).detach()
        desc2 = pred2['desc'].squeeze(0).detach()

        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8,
            device=self.device, dist='dot', block_size=2**13)

        pts3d_0 = pred1['pts3d'].squeeze(0).cpu().numpy()
        pts3d_1 = pred2['pts3d_in_other_view'].squeeze(0).cpu().numpy()

        X0 = pts3d_0[matches_im0[:,1], matches_im0[:,0]]
        X1 = pts3d_1[matches_im1[:,1], matches_im1[:,0]]
        conf = pred1['conf'].squeeze(0).cpu().numpy()[matches_im0[:,1], matches_im0[:,0]]

        # matches and Ks are in 512-resolution (resized) space
        return X0, X1, conf, matches_im0, matches_im1, K0_resize, K1_resize