
import torch
import cv2
import argparse
import numpy as np
import os
import time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
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
        
    print('Inference done.')
    
    # Sample points on a grid to reduce optimization size
    stride = 16
    trajs = traj_maps_e[:,:,:,::stride,::stride].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2
    visconfs = visconf_maps_e[:,:,:,::stride,::stride].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2
    
    pts = trajs[0].cpu().numpy() # T, N, 2
    conf = visconfs[0,:,:,1].cpu().numpy() # T, N
    
    solve_camera(pts, conf, H, W)

def project(points, camera_params, K):
    """
    Project 3D points to 2D using camera parameters.
    points: (N, 3) 3D points
    camera_params: (3, 6) or (1, 6) [rvec, tvec]
    K: (3, 3) Intrinsic matrix
    """
    # Reshape if necessary
    points = points.reshape(-1, 3)
    
    # Slice params
    rvec = camera_params[:, :3]
    tvec = camera_params[:, 3:]
    
    # Rotate
    rot = Rotation.from_rotvec(rvec)
    points_proj = rot.apply(points) + tvec
    
    # Normalize
    points_proj = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-7)
    
    # Apply K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    points_proj[:, 0] = points_proj[:, 0] * fx + cx
    points_proj[:, 1] = points_proj[:, 1] * fy + cy
    
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    points_3d_select = points_3d[point_indices]
    camera_params_select = camera_params[camera_indices]
    
    points_proj = project(points_3d_select, camera_params_select, K)
    return (points_proj - points_2d).ravel()

def bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, points_2d, K):
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    
    n = 6 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]
    
    print(f"Bundle Adjustment: {n_cameras} cameras, {n_points} points, {m//2} observations.")
    
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    
    res = least_squares(fun, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf', 
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K))
    
    return res

def solve_camera(tracks, conf, H, W):
    print("Preparing for bundle adjustment...")
    T, N, _ = tracks.shape
    
    # Intrinsics
    f = max(H, W)
    cx, cy = W/2, H/2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    # 1. Initialization: Two-View SfM (Frame 0 -> Frame T//2 or 10)
    # Filter valid points
    target_frame = min(15, T-1)
    mask_common = (conf[0] > 0) & (conf[target_frame] > 0)
    pts0 = tracks[0][mask_common]
    pts1 = tracks[target_frame][mask_common]
    
    if len(pts0) < 8:
        print("Not enough points initialization.")
        return

    E, mask_e = cv2.findEssentialMat(pts0, pts1, focal=f, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_p = cv2.recoverPose(E, pts0, pts1, focal=f, pp=(cx, cy), mask=mask_e)
    
    # Triangulate these initial points
    # Needs projection matrices P0=[I|0], P1=[R|t]
    P0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P0 = K @ P0
    P1 = np.hstack((R, t))
    P1 = K @ P1
    
    # Use only inliers from recoverPose
    valid_indices = np.where(mask_common)[0] # Indices in original N
    inlier_mask = (mask_p.ravel() > 0)
    pts0_in = pts0[inlier_mask]
    pts1_in = pts1[inlier_mask]
    
    # Points 4D
    pts4d = cv2.triangulatePoints(P0, P1, pts0_in.T, pts1_in.T)
    pts3d = pts4d[:3] / pts4d[3]
    pts3d = pts3d.T # (M, 3)
    
    # Store global 3D structure
    # Map from original track index (0..N) to optimized 3D point index
    track_to_3d_idx = {} 
    valid_track_indices = valid_indices[inlier_mask]
    
    points_3d_opt = []
    for i, track_idx in enumerate(valid_track_indices):
        track_to_3d_idx[track_idx] = i
        points_3d_opt.append(pts3d[i])
    points_3d_opt = np.array(points_3d_opt)
    
    # 2. PnP for all other cameras
    camera_params_opt = np.zeros((T, 6)) # (rvec, tvec)
    
    # Set init cameras
    camera_params_opt[0] = np.zeros(6) # Identity
    rvec_t, _ = cv2.Rodrigues(R)
    camera_params_opt[target_frame] = np.hstack((rvec_t.ravel(), t.ravel()))
    
    # We need to initialize other cameras. 
    # For "simple" script, let's assume linear interpolation or just solve PnP for each frame individually
    print("Initializing camera poses via PnP...")
    for t_idx in range(T):
        if t_idx == 0 or t_idx == target_frame:
            continue
            
        # Find 2D-3D correspondences for this frame
        # We need points that are in our points_3d_opt AND visible in frame t_idx
        
        visible_mask = conf[t_idx] > 0
        relevant_indices = [idx for idx in valid_track_indices if visible_mask[idx]]
        
        if len(relevant_indices) < 6:
            camera_params_opt[t_idx] = camera_params_opt[t_idx-1]
            continue
            
        # Get 3D points
        cur_3d_indices = [track_to_3d_idx[idx] for idx in relevant_indices]
        cur_obj_pts = points_3d_opt[cur_3d_indices]
        cur_img_pts = tracks[t_idx][relevant_indices]
        
        # Solve PnP
        success, rvec, tvec, _ = cv2.solvePnPRansac(cur_obj_pts, cur_img_pts, K, None, iterationsCount=100, reprojectionError=2.0)
        
        if success:
            camera_params_opt[t_idx] = np.hstack((rvec.ravel(), tvec.ravel()))
        else:
            camera_params_opt[t_idx] = camera_params_opt[t_idx-1]
            
    # 3. Construct BA Data Structures
    cam_idxs = []
    pt_idxs = []
    pts_2d_flat = []
    
    # Iterate all observations
    for trk_idx, pt3d_idx in track_to_3d_idx.items():
        for t_idx in range(T):
            if conf[t_idx][trk_idx] > 0:
                cam_idxs.append(t_idx)
                pt_idxs.append(pt3d_idx)
                pts_2d_flat.append(tracks[t_idx][trk_idx])
                
    cam_idxs = np.array(cam_idxs)
    pt_idxs = np.array(pt_idxs)
    pts_2d_flat = np.array(pts_2d_flat)
    
    # 4. Run BA
    print("Running Bundle Adjustment...")
    res = bundle_adjustment(camera_params_opt, points_3d_opt, cam_idxs, pt_idxs, pts_2d_flat, K)
    
    print("\nOptimization Success:", res.success)
    print("Final Cost:", res.cost)
    
    # Save or Print Trajectory
    opt_cameras = res.x[:T*6].reshape((T, 6))
    
    print("\nOptimized Trajectory (First 5 frames):")
    for i in range(min(5, T)):
        print(f"Frame {i}: t={opt_cameras[i, 3:]}")
        
    np.savetxt("trajectory.txt", opt_cameras)
    np.savetxt("structure.txt", res.x[T*6:].reshape((-1, 3)))
    print("\nSaved trajectory.txt and structure.txt")

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
    model.load_state_dict(state_dict["model"], strict=True)
    model.cuda().eval()
    
    run(model, args)
