
import torch
import cv2
import argparse
import numpy as np
import os
import time
from nets.alltracker import Net
import utils.basic
import utils.improc

def create_synthetic_video(filename, T=30, H=384, W=512):
    # Create a video with moving dots
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 10, (W, H))
    
    # 3D points
    np.random.seed(42)
    points_3d = np.random.rand(200, 3) * 10 - 5
    points_3d[:, 2] = points_3d[:, 2] + 15 # Shift away from camera
    
    # Camera motion: Translating in X (parallax)
    for t in range(T):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Simple pinhole: f = W
        f = W
        K = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]])
        
        # Move camera
        points_t = points_3d.copy()
        points_t[:, 0] -= t * 0.1 # Move points left (camera moves right)
        
        # Filter points behind camera
        valid = points_t[:, 2] > 0.1
        points_t = points_t[valid]
        
        proj = (K @ points_t.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        
        for p in proj:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < W and 0 <= y < H:
                 cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
        
        writer.write(img)
    writer.release()
    print(f"Created synthetic video {filename}")

def read_mp4(name_path):
    vidcap = cv2.VideoCapture(name_path)
    framerate = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
    frames = []
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    vidcap.release()
    return frames, framerate

def run(model, args):
    rgbs, framerate = read_mp4(args.mp4_path)
    if not rgbs:
        print("Failed to load video")
        return

    H,W = rgbs[0].shape[:2]
    scale = min(int(args.image_size)/H, int(args.image_size)/W)
    H, W = int(H*scale), int(W*scale)
    # Resize to div 8
    H_ = H//8 * 8
    W_ = W//8 * 8
    rgbs = [cv2.resize(rgb, dsize=(W_, H_), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
    
    rgbs_t = [torch.from_numpy(rgb).permute(2,0,1) for rgb in rgbs]
    rgbs_t = torch.stack(rgbs_t, dim=0).unsqueeze(0).float().cuda() # 1,T,C,H,W
    
    B,T,C,H,W = rgbs_t.shape
    
    grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float()
    grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W)

    print('Starting inference...')
    with torch.no_grad():
        flows_e, visconf_maps_e, _, _ = \
            model.forward_sliding(rgbs_t[:, args.query_frame:], iters=4, sw=None, is_training=False)
        traj_maps_e = flows_e.cuda() + grid_xy # B,Tf,2,H,W
        
        # Handle backward if needed (simplified to forward only for demo)
        
    print('Inference done.')
    
    # Sample points on a grid
    stride = 16
    trajs = traj_maps_e[:,:,:,::stride,::stride].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2
    visconfs = visconf_maps_e[:,:,:,::stride,::stride].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2
    
    pts = trajs[0].cpu().numpy() # T, N, 2
    conf = visconfs[0,:,:,1].cpu().numpy() # T, N (confidence)
    vis = torch.sigmoid(visconfs[0,:,:,0]).cpu().numpy() # T, N (visibility)
    
    solve_camera(pts, conf, H, W)

def solve_camera(tracks, conf, H, W):
    # tracks: T, N, 2
    T, N, _ = tracks.shape
    
    # Use standard intrinsics approx
    f = max(H, W)
    pp = (W/2, H/2)
    K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]])
    
    # Using Frame 0 and Frame 10 (or last)
    # Filter points visible in both
    target_frame = min(10, T-1)
    
    mk0 = conf[0] > 0
    mk1 = conf[target_frame] > 0
    # Actually confidence is usually logits in some codes, but here we took raw output? 
    # forward_sliding returns visconf_maps_e which are raw logits usually?
    # Actually in demo.py line 161: visconfs_e[0,:,:,1] > args.conf_thr
    # So valid points check:
    
    valid = (conf[0] > 0) & (conf[target_frame] > 0)
    
    pts1 = tracks[0][valid]
    pts2 = tracks[target_frame][valid]
    
    if len(pts1) < 8:
        print("Not enough points for Essential Matrix")
        return
        
    print(f"Using {len(pts1)} points for solve between frame 0 and frame {target_frame}")
    
    E, mask = cv2.findEssentialMat(pts1, pts2, focal=f, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inliers = mask.sum() if mask is not None else 0
    print(f"Found Essential Matrix with {inliers} / {len(pts1)} inliers")
    
    if inliers < 5:
        print("Too few inliers")
        return

    points, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, focal=f, pp=pp, mask=mask)
    
    print("Estimated Rotation:\n", R)
    print("Estimated Translation (direction):\n", t)
    print("Success! Camera solve attempted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp4_path", type=str, default='synthetic_dots.mp4')
    parser.add_argument("--image_size", type=int, default=1024) # max dimension of a video frame (upsample to this)
    parser.add_argument("--query_frame", type=int, default=0)
    args = parser.parse_args()
    
    if not os.path.exists(args.mp4_path):
        create_synthetic_video(args.mp4_path)

    # Load Tiny model for speed/memory
    model = Net(16, use_basicencoder=True, no_split=True)
    # Load weights
    state_dict = torch.load("checkpoints/alltracker_tiny.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.cuda().eval()
    
    run(model, args)
