import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import torch
import cv2
import argparse
import numpy as np
import glob
import re
import utils.saveload
import utils.basic
import utils.improc
from nets.alltracker import Net
import time
from concurrent.futures import ThreadPoolExecutor

def parse_pattern(pattern):
    """
    Parses input patterns like file.####.exr or file.%04d.exr
    Returns a glob pattern to find files, and a printf-style pattern for output/parsing.
    """
    if '%' in pattern:
        glob_pat = re.sub(r'%0?\d*d', '*', pattern)
        return glob_pat, pattern
    
    match = re.search(r'(#+)', pattern)
    if match:
        n = len(match.group(1))
        printf_pat = pattern.replace(match.group(1), f'%0{n}d')
        glob_pat = pattern.replace(match.group(1), '*')
        return glob_pat, printf_pat
    
    return pattern, pattern

def extract_frame_num(filename):
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else 0

def process_single_exr(f):
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None

    if len(im.shape) == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    elif im.shape[2] == 4:
        im = im[:, :, :3]
    
    # Convert linear EXR to sRGB for the model
    im = np.clip(im, 0.0, 1.0)
    im = np.power(im, 1.0/2.2)
    im = (im * 255).astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def read_exr_sequence(pattern, args):
    glob_pat, _ = parse_pattern(pattern)
    files = sorted(glob.glob(glob_pat))
    
    if not files:
        print(f"No files found matching: {glob_pat}")
        return [], None, []

    print(f"Found {len(files)} files. Loading...")
    t0 = time.time()
    
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(process_single_exr, files))

    frames = []
    original_hw = None
    frame_nums = []

    for i, im in enumerate(images):
        if im is not None:
            frames.append(im)
            frame_nums.append(extract_frame_num(files[i]))
            if original_hw is None:
                original_hw = im.shape[:2]
    
    print(f"Loaded {len(frames)} frames in {time.time()-t0:.2f}s")
    return frames, original_hw, frame_nums

def forward_sequence(rgbs, model, args):
    B, T, C, H, W = rgbs.shape
    grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float()
    grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)

    # Forward tracking
    flows_e, visconf_maps_e, _, _ = model.forward_sliding(
        rgbs[:, args.query_frame:], iters=args.inference_iters, is_training=False
    )
    traj_maps_e = flows_e.cuda() + grid_xy

    # Backward tracking
    if args.query_frame > 0:
        backward_flows_e, backward_visconf_maps_e, _, _ = model.forward_sliding(
            rgbs[:, :args.query_frame+1].flip([1]), iters=args.inference_iters, is_training=False
        )
        b_traj = (backward_flows_e.cuda() + grid_xy).flip([1])[:, :-1]
        b_vis = backward_visconf_maps_e.flip([1])[:, :-1]
        traj_maps_e = torch.cat([b_traj, traj_maps_e], dim=1)
        visconf_maps_e = torch.cat([b_vis, visconf_maps_e], dim=1)

    # Normalize to UV (0-1) and extract visibility mask
    trajs_uv = traj_maps_e.clone()
    trajs_uv[:, :, 0] /= (W - 1)
    trajs_uv[:, :, 1] /= (H - 1)
    vis_mask = torch.sigmoid(visconf_maps_e[:, :, 1])
    
    return trajs_uv, vis_mask

    print(f"Saved {trajs_cpu.shape[0]} EXR files in {time.time()-t0:.2f}s")

def compute_inverse_map(trajs, vis, H, W):
    """
    Computes an inverse map (Matchmove) from dense forward tracks using splatting + inpainting.
    trajs: (N, 2) normalized UV coordinates where pixels from Frame 0 went.
    vis: (N,) visibility confidence.
    H, W: Target dimensions.
    Returns: (H, W, 2) Inverse UV map.
    """
    # Create the source grid (Frame 0 UVs)
    H_src, W_src = int(np.sqrt(trajs.shape[0]*H/W)), int(np.sqrt(trajs.shape[0]*W/H)) # Approx
    # Actually, trajs is flattened from the original grid.
    # We can reconstruct simple UVs for the source indices.
    N = trajs.shape[0]
    
    # We assume the source was a regular grid.
    # We can just generate 0..1 values for the source indices.
    # Since we don't know the exact aspect of source grid if it was reshaped, 
    # we can try to infer or just pass it? 
    # Actually, in run(), we know the model input size.
    # Let's just generate the grid on the fly in the main loop or pass indices.
    pass

def run(model, args):
    rgbs_list, original_hw, frame_nums = read_exr_sequence(args.input, args)
    if not rgbs_list: return

    H_orig, W_orig = original_hw
    scale = min(args.image_size / H_orig, args.image_size / W_orig)
    H_model, W_model = int(H_orig * scale) // 8 * 8, int(W_orig * scale) // 8 * 8
    
    print(f"Resizing to {W_model}x{H_model}...")
    t0 = time.time()
    def resize_one(rgb):
        return cv2.resize(rgb, (W_model, H_model))
    
    with ThreadPoolExecutor() as executor:
        rgbs_resized = list(executor.map(resize_one, rgbs_list))
    print(f"Resized in {time.time()-t0:.2f}s")

    rgbs_tensor = torch.stack([torch.from_numpy(r).permute(2,0,1) for r in rgbs_resized]).unsqueeze(0).float().cuda()

    print("Running inference...")
    t0 = time.time()
    with torch.no_grad():
        trajs_uv, vis_mask = forward_sequence(rgbs_tensor, model, args)
    print(f"Inference finished in {time.time()-t0:.2f}s")

    _, printf_pat = parse_pattern(args.output)
    out_dir = os.path.dirname(printf_pat)
    if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)

    trajs_cpu = trajs_uv[0].cpu().numpy()
    vis_cpu = vis_mask[0].cpu().numpy()

    print("Saving EXR files...")
    t0 = time.time()
    
    # Pre-compute source grid UVs for matchmove
    if args.matchmove:
        # trajs_cpu shape is (T, 2, H, W)
        # We want to create a source grid matching H, W
        H_map, W_map = trajs_cpu.shape[2], trajs_cpu.shape[3]
        gy, gx = np.meshgrid(np.linspace(0, 1, H_map), np.linspace(0, 1, W_map), indexing='ij')
        # Flattened source UVs: (N, 2)
        source_uv_flat = np.stack([gx.flatten(), gy.flatten()], axis=1).astype(np.float32)

    def save_one(t):
        if args.matchmove:
            # Inverse Map Generation (Splatting)
            # trajs_cpu[t] is (2, H, W)
            H_map, W_map = trajs_cpu.shape[2], trajs_cpu.shape[3]
            
            # Destination coordinates (where the pixels form Frame 0 went)
            # trajs_cpu[t] is (2, H, W) -> transpose to (H, W, 2)
            dest_uv = trajs_cpu[t].transpose(1, 2, 0).reshape(-1, 2)
            
            # Source coordinates (their original identity) are source_uv_flat
            
            # Filter by visibility to reduce noise
            valid = vis_cpu[t].reshape(-1) > 0.5
            
            dest_x = (dest_uv[valid, 0] * (W_map - 1)).astype(np.int32)
            dest_y = (dest_uv[valid, 1] * (H_map - 1)).astype(np.int32)
            
            # Clip
            dest_x = np.clip(dest_x, 0, W_map - 1)
            dest_y = np.clip(dest_y, 0, H_map - 1)
            
            # Initialize Inverse Map
            # Channels: U, V, Weight
            inv_map = np.zeros((H_map, W_map, 3), dtype=np.float32)
            
            # Splat (Latest write wins)
            # Nuke expects 0,0 BL, so invert V of the SOURCE value
            vals = source_uv_flat[valid].copy()
            vals[:, 1] = 1.0 - vals[:, 1]
            
            inv_map[dest_y, dest_x, 0] = vals[:, 0] # U
            inv_map[dest_y, dest_x, 1] = vals[:, 1] # V (Inverted)
            inv_map[dest_y, dest_x, 2] = 1.0        # Mask
            
            # Hole Filling (Inpainting)
            mask = (inv_map[:, :, 2] == 0).astype(np.uint8)
            
            # If mask is empty, we are good. If mask is full, we failed.
            if np.sum(mask) > 0 and np.sum(mask) < mask.size:
                # cv2.inpaint requires 8-bit or 16-bit input image? 
                # Actually inpaint works on 8-bit, 16-bit unsigned, 32-bit float.
                # But it expects the IMAGE to be one of those.
                inv_map[:, :, 0] = cv2.inpaint(inv_map[:, :, 0], mask, 3, cv2.INPAINT_TELEA)
                inv_map[:, :, 1] = cv2.inpaint(inv_map[:, :, 1], mask, 3, cv2.INPAINT_TELEA)
                # For alpha, we might want to keep it soft or binary?
                # If we inpaint alpha, it becomes 1 everywhere.
                # For matchmove, we usually want alpha=0 where there is NO surface.
                # But inpainting fills the holes...
                # Let's set alpha to 1 where we filled holes.
                inv_map[:, :, 2] = 1.0 - (mask/255.0) # This logic is circular.
                # Actually, strictly, the "Texture" should only appear where the object truly IS.
                # Inpainting fills *cracks* but also *large regions* if we let it.
                # We should probably only inpaint small cracks.
                # But for now, full inpaint is safer visually than holes.
                inv_map[:, :, 2] = 1.0 # Assuming we filled everything
            
            out_exr = np.dstack([np.zeros_like(inv_map[:,:,0]), inv_map[:,:,1], inv_map[:,:,0], inv_map[:,:,2]]).astype(np.float32)
        
        else:
            # Standard Stabilize Map
            uv_map = trajs_cpu[t].transpose(1, 2, 0)
            alpha = vis_cpu[t]
            
            # Nuke STMap: R=U, G=V, B=0, A=Mask
            # OpenCV imwrite BGR: B=0, G=V, R=U, A=Mask
            # Invert V for Nuke (bottom-left 0,0), so we use 1.0 - V
            out_exr = np.dstack([np.zeros_like(alpha), 1.0 - uv_map[:,:,1], uv_map[:,:,0], alpha]).astype(np.float32)
            
        cv2.imwrite(printf_pat % frame_nums[t], out_exr)

    with ThreadPoolExecutor() as executor:
        list(executor.map(save_one, range(trajs_cpu.shape[0])))
    
    print(f"Saved {trajs_cpu.shape[0]} EXR files in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--ckpt_init", type=str, default='')
    parser.add_argument("--query_frame", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--inference_iters", type=int, default=4)
    parser.add_argument("--window_len", type=int, default=16)
    parser.add_argument("--tiny", action='store_true')
    parser.add_argument("--matchmove", action='store_true', help="Output inverse STMap (Texture -> Footage) instead of Stabilize Map")
    args = parser.parse_args()
    
    model = Net(args.window_len, use_basicencoder=args.tiny, no_split=args.tiny)
    if args.ckpt_init:
        utils.saveload.load(None, args.ckpt_init, model)
    else:
        url = f"https://huggingface.co/aharley/alltracker/resolve/main/alltracker{'_tiny' if args.tiny else ''}.pth"
        model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location='cpu')['model'])
    
    model.cuda().eval()
    run(model, args)