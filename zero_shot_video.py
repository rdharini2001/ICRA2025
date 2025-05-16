# Author: Srijan Dokania
# Description: This script runs the Zero-Shot Pose Estimation on video frames using CLIPSeg and MiDaS.
# It uses Monte Carlo sampling for segmentation and computes the pose using PCA.
# It also provides options for camera intrinsics, segmentation threshold, and output video settings.

#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import json
from tqdm import tqdm
import glob
import os
import gc
def parse_args():
    p = argparse.ArgumentParser(
        description="Zero‑Shot Pose Estimation via MC‑CLIPSeg + MiDaS + Weighted PCA"
    )
    p.add_argument("--frames-folder", required=True,
                   help="Path to folder containing input RGB image frames (e.g., frame000001.jpg, ...).")
    p.add_argument("--output-video",  default="pose_output.mp4",
                   help="Path to save the output video file.")
    p.add_argument("--image", default="img.png", help="Path to input RGB image.")
    p.add_argument("--prompt",    default="robot",
                   help="Text prompt for segmentation (e.g. 'robot').")
    p.add_argument("--fx",   type=float, default=422.4035339355469,
                   help="Camera focal length in px (x).")
    p.add_argument("--fy",   type=float, default=421.90533447265625,
                   help="Camera focal length in px (y).")
    p.add_argument("--cx",   type=float, default=417.8716735839844,
                   help="Camera principal point x (default=width/2).")
    p.add_argument("--cy",   type=float, default=244.17526245117188,
                   help="Camera principal point y (default=height/2).")
    p.add_argument("--depth-model", default="MiDaS",
                   choices=["MiDaS_small","MiDaS"], help="MiDaS variant.")
    p.add_argument("--mc-runs",     type=int,   default=5,
                   help="Monte Carlo runs for dropout segmentation.")
    p.add_argument("--mask-thresh", type=float, default=0.6,
                   help="Threshold on avg. seg confidence for binary mask.")
    p.add_argument("--fps",         type=float, default=15.0,
                   help="Frames per second for the output video.")
    p.add_argument("--codec",       default="MP4V",
                   help="FourCC codec for the output video (e.g., MP4V, XVID).")

    return p.parse_args()

def segment_with_mc_clipseg(img, prompt, model, processor, device="cpu", mc_runs=20):
    """
    Run CLIPSeg with dropout mc_runs times, average into a per-pixel confidence map.
    Handles both 3D and 4D logits outputs.
    """
    model.train()  # turn on dropout layers

    H, W = img.shape[:2]
    conf_map = torch.zeros((H, W), device=device)

    for _ in range(mc_runs):
        # ensure dropout is active
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)
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
    print(f"Avg. confidence: {conf_map.mean():.3f} (min: {conf_map.min():.3f}, max: {conf_map.max():.3f})")
    return conf_map.cpu().detach().numpy()

def predict_depth(img, midas_model, midas_transform, device="cpu"):
    """Metric depth (0–5 m) via MiDaS."""
    midas_model.eval()
    inp = midas_transform(img).to(device)
    if inp.ndim == 3:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        out = midas_model(inp)
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

    colors = [(0,0,255),(0,255,0),(255,0,0)] # BGR: Red(X), Green(Y), Blue(Z)
    for i, end in enumerate(ends2d):
        cv2.arrowedLine(img, tuple(center2d), tuple(end),
                        colors[i], 2, tipLength=0.1)
        cv2.circle(img, tuple(end), 3, colors[i], -1)

    cv2.circle(img, tuple(center2d), 4, (255,255,255), -1)
    cv2.circle(img, tuple(center2d), 2, (0,0,0), -1)

def main():
    args = parse_args()
    # --- 0. Setup ---
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load models ONCE ---
    print("Loading CLIPSeg model...")
    clip_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)
    clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    print("Loading MiDaS model...")
    midas_model = torch.hub.load("intel-isl/MiDaS", args.depth_model).to(device)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transform = midas_transforms.small_transform if args.depth_model == "MiDaS_small" else midas_transforms.default_transform
    print("Models loaded.")

     # --- Get list of image frames ---
    image_files = sorted(glob.glob(os.path.join(args.frames_folder, '*.jpg')) + \
                         glob.glob(os.path.join(args.frames_folder, '*.png')))
    if not image_files:
        print(f"Error: No image files (.jpg, .png) found in {args.frames_folder}")
        return

    # --- Initialize Video Writer ---
    # Get frame size from the first image
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Error: Could not read first frame {image_files[0]}")
        return
    h, w = first_frame.shape[:2]
    frame_size = (w, h)
    print(f"Frame size: {frame_size}")
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    video_writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, frame_size)
    print(f"Output video: {args.output_video} ({w}x{h} @ {args.fps}fps, Codec: {args.codec})")
   
    # --- Process each frame ---
    for frame_path in tqdm(image_files, desc="Processing frames"):
        i = 0
        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            print(f"Warning: Could not read frame {frame_path}, skipping.")
            continue
        if img_bgr.shape[0] != h or img_bgr.shape[1] != w:
            print(f"Warning: Frame {frame_path} has different size ({img_bgr.shape[1]}x{img_bgr.shape[0]}), skipping.")
            continue # Skip frames with inconsistent sizes

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- Get camera intrinsics for the current frame ---
        fx = args.fx
        fy = args.fy
        cx = args.cx if args.cx is not None else w/2
        cy = args.cy if args.cy is not None else h/2

        # 1) Monte Carlo segmentation
        print("[1/5] Running MC CLIPSeg segmentation…")
        seg_conf = segment_with_mc_clipseg(
            img_rgb, args.prompt,model=clip_model, processor=clip_processor,
            device=device,
            mc_runs=args.mc_runs
        )
        # cv2.imshow("Segmentation Confidence", (seg_conf*255).astype(np.uint8))
        # cv2.waitKey(1)

        # binary mask
        mask = (seg_conf > args.mask_thresh).astype(np.uint8)
        # cv2.imshow("Binary Mask", mask*255)
        # cv2.waitKey(1)

        # 2) Depth estimation
        print("[2/5] Estimating depth…")
        depth = predict_depth(
            img_rgb, midas_model=midas_model,
            midas_transform=midas_transform,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        disp = np.uint8((depth / depth.max()) * 255)
        # cv2.imshow("Depth Map", disp)
        # cv2.waitKey(1)

        # 3) Weighted point cloud
        print("[3/5] Unprojecting to weighted point cloud…")
        pts3d, weights = weighted_unproject(depth, seg_conf, fx, fy, cx, cy)

        # 4) Weighted PCA + covariance
        print("[4/5] Computing weighted PCA & covariance…")
        if len(pts3d) < 3:
            print(f"Warning: Not enough points ({len(pts3d)}) after unprojection for frame {frame_path}, skipping PCA.")
            out = img_bgr.copy() # Use original frame for output
        else:
            # 4) Weighted PCA + covariance
            try:
                Wsum = weights.sum()
                if Wsum < 1e-6: # Avoid division by zero if all weights are tiny
                    raise ValueError("Sum of weights is too small.")

                centroid = (weights[:,None] * pts3d).sum(axis=0) / Wsum

                diffs = pts3d - centroid
                cov   = (weights[:,None,None] * (diffs[:,:,None] * diffs[:,None,:])).sum(axis=0) / Wsum

                eigvals, eigvecs = np.linalg.eigh(cov)
                idx = np.argsort(eigvals)[::-1]
                # eigvals = eigvals[idx] # Not used later, commented out for efficiency
                axes    = eigvecs[:,idx]
                print("Cov eigenvalues:", eigvals)
                print("3σ radii (m):", np.sqrt(eigvals)*3)
                # Ensure axes form a right-handed coordinate system (optional but good practice)
                if np.linalg.det(axes) < 0:
                    axes[:, -1] *= -1

                # 5) Draw and save
                print("[5/5] Drawing axes & saving results…")
                out = img_bgr.copy()
                draw_axes_and_keypoints(out, centroid, axes, fx, fy, cx, cy)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, cnts, -1, (0,255,255), 2)
            except Exception as e:
                print(f"Warning: Error during PCA or drawing for frame {frame_path}: {e}. Using original frame.")
                out = img_bgr.copy()
        # Write frame to video
        video_writer.write(out)
        i += 1
        # --- Optional: Clear CUDA cache periodically if memory is tight ---
        if device == "cuda":
            if i % 10 == 0: # Example: clear every 10 frames
                torch.cuda.empty_cache()
                gc.collect()

    # --- Release Video Writer ---
    video_writer.release()
    print(f"Finished processing. Video saved to {args.output_video}")


if __name__ == "__main__":
    main()


