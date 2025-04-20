#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def parse_args():
    p = argparse.ArgumentParser(
        description="Zero‑Shot Pose Estimation via MC‑CLIPSeg + MiDaS + Weighted PCA"
    )
    p.add_argument("image", help="Path to input RGB image.")
    p.add_argument("--prompt",    default="robot",
                   help="Text prompt for segmentation (e.g. 'robot').")
    p.add_argument("--fx",   type=float, default=500.0,
                   help="Camera focal length in px (x).")
    p.add_argument("--fy",   type=float, default=500.0,
                   help="Camera focal length in px (y).")
    p.add_argument("--cx",   type=float,
                   help="Camera principal point x (default=width/2).")
    p.add_argument("--cy",   type=float,
                   help="Camera principal point y (default=height/2).")
    p.add_argument("--depth-model", default="MiDaS_small",
                   choices=["MiDaS_small","MiDaS"], help="MiDaS variant.")
    p.add_argument("--mc-runs",     type=int,   default=20,
                   help="Monte Carlo runs for dropout segmentation.")
    p.add_argument("--mask-thresh", type=float, default=0.5,
                   help="Threshold on avg. seg confidence for binary mask.")
    return p.parse_args()

def segment_with_mc_clipseg(img, prompt, device="cpu", mc_runs=20):
    """
    Run CLIPSeg with dropout mc_runs times, average into a per-pixel confidence map.
    Handles both 3D and 4D logits outputs.
    """
    proc  = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained(
                "CIDAS/clipseg-rd64-refined").to(device)
    model.train()  # turn on dropout layers

    H, W = img.shape[:2]
    conf_map = torch.zeros((H, W), device=device)

    for _ in range(mc_runs):
        # ensure dropout is active
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        inputs = proc(text=[prompt], images=[img], return_tensors="pt").to(device)
        raw_logits = model(**inputs).logits  # could be [B,1,h,w] or [B,h,w]
        # extract a 2D logits map:
        if raw_logits.ndim == 4:
            logits = raw_logits[0,0]     # [h_small, w_small]
        elif raw_logits.ndim == 3:
            logits = raw_logits[0]       # [h_small, w_small]
        else:
            raise ValueError(f"Unexpected logits dims: {raw_logits.shape}")

        prob = torch.sigmoid(logits)       # [0,1] at low res

        # upsample to full resolution:
        prob_up = F.interpolate(
            prob.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="nearest"
        ).squeeze()
        conf_map += prob_up

    conf_map /= float(mc_runs)
    return conf_map.cpu().detach().numpy()

def predict_depth(img, variant, device="cpu"):
    """Metric depth (0–5 m) via MiDaS."""
    midas = torch.hub.load("intel-isl/MiDaS", variant).to(device)
    midas.eval()
    trans = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = trans.small_transform if variant=="MiDaS_small" else trans.default_transform

    inp = transform(img).to(device)
    if inp.ndim == 3:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        out = midas(inp)
    depth = out.squeeze().cpu().detach().numpy()
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    depth = (depth - depth.min())/(depth.max()-depth.min()+1e-8)
    return depth * 5.0

def weighted_unproject(depth, conf, fx, fy, cx, cy):
    """Backproject pixels to 3D points, weighted by conf[y,x]."""
    h, w = depth.shape
    ys, xs = np.indices((h, w))
    zs = depth
    xs = (xs - cx) * zs / fx
    ys = (ys - cy) * zs / fy
    pts = np.stack([xs, ys, zs], axis=-1).reshape(-1,3)
    ws  = conf.reshape(-1)
    valid = ws > 1e-3
    return pts[valid], ws[valid]

def draw_axes_and_keypoints(img, centroid, axes, fx, fy, cx, cy, scale=0.5):
    """Project and draw XYZ axes from centroid."""
    length = scale * np.median(centroid[2])
    c = centroid.reshape(1,3)
    ends = c + axes * length

    def project(pts):
        x,y,z = pts[:,0], pts[:,1], pts[:,2]
        u = (x * fx / z) + cx
        v = (y * fy / z) + cy
        return np.stack([u, v], axis=-1)

    center2d = project(c).astype(int)[0]
    ends2d   = project(ends).astype(int)

    colors = [(0,0,255),(0,255,0),(255,0,0)]
    for i, end in enumerate(ends2d):
        cv2.arrowedLine(img, tuple(center2d), tuple(end),
                        colors[i], 3, tipLength=0.1)
        cv2.circle(img, tuple(end), 5, colors[i], -1)

    cv2.circle(img, tuple(center2d), 6, (255,255,255), -1)
    cv2.circle(img, tuple(center2d), 4, (0,0,0), -1)

def main():
    args = parse_args()
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    fx = args.fx
    fy = args.fy
    cx = args.cx if args.cx is not None else w/2
    cy = args.cy if args.cy is not None else h/2

    # 1) Monte Carlo segmentation
    print("[1/5] Running MC CLIPSeg segmentation…")
    seg_conf = segment_with_mc_clipseg(
        img_rgb, args.prompt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mc_runs=args.mc_runs
    )
    cv2.imshow("Segmentation Confidence", (seg_conf*255).astype(np.uint8))
    cv2.waitKey(1)

    # binary mask
    mask = (seg_conf > args.mask_thresh).astype(np.uint8)
    cv2.imshow("Binary Mask", mask*255)
    cv2.waitKey(1)

    # 2) Depth estimation
    print("[2/5] Estimating depth…")
    depth = predict_depth(
        img_rgb, args.depth_model,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    disp = np.uint8((depth / depth.max()) * 255)
    cv2.imshow("Depth Map", disp)
    cv2.waitKey(1)

    # 3) Weighted point cloud
    print("[3/5] Unprojecting to weighted point cloud…")
    pts3d, weights = weighted_unproject(depth, seg_conf, fx, fy, cx, cy)

    # 4) Weighted PCA + covariance
    print("[4/5] Computing weighted PCA & covariance…")
    Wsum = weights.sum()
    centroid = (weights[:,None] * pts3d).sum(axis=0) / Wsum

    diffs = pts3d - centroid
    cov   = (weights[:,None,None] * (diffs[:,:,None] * diffs[:,None,:])).sum(axis=0) / Wsum

    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    axes    = eigvecs[:,idx]

    print("Cov eigenvalues:", eigvals)
    print("3σ radii (m):", np.sqrt(eigvals)*3)

    # 5) Draw and save
    print("[5/5] Drawing axes & saving results…")
    out = img_bgr.copy()
    draw_axes_and_keypoints(out, centroid, axes, fx, fy, cx, cy)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0,255,255), 2)

    cv2.imshow("Final Pose", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("pose_output.png", out)

    print("Translation (m):", centroid)
    print("Rotation axes (columns):\n", axes)

if __name__ == "__main__":
    main()
