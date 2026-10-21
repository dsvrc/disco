import numpy
import os
import pickle
import numpy as np
from os.path import join as pjoin
import torch
import tqdm
from scipy.spatial.transform import Rotation as R
from human_body_prior.body_model.body_model import BodyModel
import smplx
import sys
from pathlib import Path
import argparse
from typing import Optional


# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.humanml.quaternion import cont6d_to_matrix_np, cont6d_to_matrix_np_finedance
import utils.humanml.motion_process as hml_motion
from utils.humanml.paramUtil import t2m_kinematic_chain, t2m_raw_offsets
from utils.visualize import render_single_video
# from utils.vis_151d.vis import SMPLSkeleton,ax_to_6v
from utils.motionstreamer.face_z_transform import face_z_transform
from utils.motionstreamer.representation_272 import extract_features as extract_272_features

def get_body_model(
    bm_path="deps/smpl/SMPL_NEUTRAL.npz",
    model_type="smpl",
    num_betas=10,
    batch_size=1,
    device=None,
):
    """Load a BodyModel; raises if the model file is missing."""
    if not os.path.exists(bm_path):
        raise FileNotFoundError(
            f"SMPL model not found at {bm_path}. Please place the neutral npz there or update the path."
        )
    comp_device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return BodyModel(
        bm_path=bm_path,
        model_type=model_type,
        num_betas=num_betas,
        batch_size=batch_size,
    ).to(comp_device)

########################## Load Motion Function #######################
def load_motion_aistpp(motion_file):
    """Load raw motion data from AIST++ pkl file."""
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
    smpl_poses = motion_data["smpl_poses"]  # (T, 72)
    smpl_scaling = motion_data["smpl_scaling"]  # float
    smpl_trans = motion_data["smpl_trans"]  # (T, 3)
    return smpl_poses, smpl_scaling, smpl_trans

def load_motion_finedance(motion_file):
    """Load raw motion data from FineDance npy file."""
    data = np.load(motion_file)
    if data.shape[1] == 315:
        pos = data[:, :3]   # length, c
        q = data[:, 3:]  # length, 52 * 6
    elif data.shape[1] == 319:
        pos = data[:, 4:7]   # length, c
        q = data[:, 7:]  # length, 52 * 6
    else:
        raise ValueError(f"Unexpected FineDance motion dimension: {data.shape[1]}")
    return pos, q



########################## Process Motion Function #######################
def process_motion_smpl_263(root_orient, pose_body, betas, trans, mirror=False, feet_thre=0.002):
    """
    Process SMPL motion into HumanML3D 263-dim format
    Reference: https://github.com/EricGuo5513/HumanML3D/tree/main 
    Args: 
        root_orient: (num_frames, 3) torch tensor of root joint orientation in axis-angle
        pose_body: (num_frames, 63) torch tensor of body joint orientations in axis-angle 
        betas: (num_frames, 10) torch tensor of shape 
        trans: (num_frames, 3) torch tensor of global translation 
        feet_thre: threshold to determine foot contact 
    Returns: 
        motion_feat: (num_frames, 263) numpy array of HumanML3D motion 
    """
    num_frames = root_orient.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bm = get_body_model(batch_size=num_frames, device=device)

    with torch.no_grad():
        body_out = bm(
            root_orient=root_orient,
            pose_body=pose_body,
            betas=betas,
            trans=trans,
        )
        joint_positions = body_out.Jtr.detach().cpu().numpy()
    joint_positions = joint_positions[:, :22, :]  # Use first 22 joints only
    # Prepare HumanML3D processing globals
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain
    face_joint_indx = [2, 1, 17, 16]
    fid_r, fid_l = [8, 11], [7, 10]
    l_idx1, l_idx2 = 5, 8
    tgt_skel = hml_motion.Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(joint_positions[0]))
    # Inject required globals for utils.humanml.motion_process.process_file
    hml_motion.n_raw_offsets = n_raw_offsets
    hml_motion.kinematic_chain = kinematic_chain
    hml_motion.face_joint_indx = face_joint_indx
    hml_motion.fid_r = fid_r
    hml_motion.fid_l = fid_l
    hml_motion.l_idx1 = l_idx1
    hml_motion.l_idx2 = l_idx2
    hml_motion.tgt_offsets = tgt_offsets

    # Generate 263-dim HumanML3D features
    motion_feat, _, _, _ = hml_motion.process_file(joint_positions, feet_thre=feet_thre, mirror=mirror)
    return motion_feat

def process_motion_smpl_272(root_orient, pose_body, betas, trans, mirror=False):
    """
    process SMPL motion into MotionStreamer 272-dim format
    Reference: https://github.com/Li-xingXiao/272-dim-Motion-Representation
    Args: 
        root_orient: (num_frames, 3) torch tensor of root joint orientation in axis-angle 
        pose_body: (num_frames, 63) torch tensor of body joint orientations in axis-angle 
        betas: (num_frames, 10) torch tensor of shape 
        trans: (num_frames, 3) torch tensor of global translation 
    Returns: 
        motion_feat: (num_frames, 272) numpy array of MotionStreamer motion
    """
    num_frames = root_orient.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bm = get_body_model(batch_size=num_frames, device=device)

    # face z+ transform
    pose_aa = torch.cat([root_orient, pose_body], dim=1).reshape(num_frames, 22, 3)
    pose_aa, trans, betas = face_z_transform(trans.cpu().numpy(), betas.cpu().numpy(), pose_aa.cpu().numpy())

    # check mirror
    if mirror:
        pose_aa, trans = hml_motion.swap_left_right_smpl(pose_aa, trans)

    # get global joint positions
    pose_aa = torch.from_numpy(pose_aa).float().to(device)
    trans = torch.from_numpy(trans).float().to(device)
    betas = torch.from_numpy(betas).float().to(device)
    with torch.no_grad():
        body_out = bm(
            root_orient=pose_aa[:,:3],
            pose_body=pose_aa[:,3:],
            betas=betas,
            trans=trans,
        )
        joint_positions = body_out.Jtr.detach().cpu().numpy()
    joint_positions = joint_positions[:, :22, :]

    # Generate 272-dim MotionStreamer features
    motion_feat = extract_272_features(joint_positions, pose_aa.cpu().numpy())
    return motion_feat


def process_motion_aistpp(motion_file, tgt_fps=30, rep='263', mirror=False):
    """
    Process AIST++ motion into HumanML3D 263-dim format.
    """
    # Load raw SMPL parameters
    poses, scale, trans = load_motion_aistpp(motion_file)

    # Normalize translation and roughly floor-align height
    trans = trans.astype(numpy.float32) / float(scale)
    trans[:, 1] -= 1.0

    # Downsample from AIST++ native 60fps to target fps
    src_fps = 60
    if tgt_fps <= 0 or src_fps % tgt_fps != 0:
        raise ValueError(f"Unsupported target fps: {tgt_fps}. Expected a divisor of {src_fps}.")
    ds = src_fps // tgt_fps
    poses = poses[::ds]
    trans = trans[::ds]

    # Prepare body model
    num_frames = poses.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bm = get_body_model(batch_size=num_frames, device=device)

    # human_body_prior BodyModel expects root_orient (B,3) and pose_body ((J-1)*3)
    root_orient = torch.from_numpy(poses[:, :3]).float().to(device)
    pose_body = torch.from_numpy(poses[:, 3:66]).float().to(device)
    trans_t = torch.from_numpy(trans).float().to(device)
    betas = torch.zeros((num_frames, 10), device=device)

    if rep == '263':
        # Generate 263-dim HumanML3D features
        motion_feat = process_motion_smpl_263(
            root_orient=root_orient,
            pose_body=pose_body,
            betas=betas,
            trans=trans_t,
            mirror=mirror
        )
    elif rep == '272':
        # Generate 272-dim MotionStreamer features
        motion_feat = process_motion_smpl_272(
            root_orient=root_orient,
            pose_body=pose_body,
            betas=betas,
            trans=trans_t,
            mirror=mirror
        )
    else:
        raise ValueError(f"Unsupported representation: {rep}. Choose '263' or '272'.")
    return motion_feat

def process_motion_finedance(motion_file, tgt_fps=30, rep='263', mirror=False):
    """
    Process FineDance motion into HumanML3D 263-dim format.
    """
    # Load raw FineDance motion
    pos, q = load_motion_finedance(motion_file)

    # Downsample from FineDance native 60fps to target fps
    src_fps = 30
    if tgt_fps <= 0 or src_fps % tgt_fps != 0:
        raise ValueError(f"Unsupported target fps: {tgt_fps}. Expected a divisor of {src_fps}.")
    ds = src_fps // tgt_fps
    pos = pos[::ds]
    q = q[::ds]

    num_frames = q.shape[0]
    joints_num = q.shape[1] // 6
    # Prepare body model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert 6D rotations to axis-angle
    q_reshaped = q.reshape(-1, 6)
    # Convert 6D to axis-angle using scipy
    rot_mats = cont6d_to_matrix_np_finedance(q_reshaped)
    axis_angles = R.from_matrix(rot_mats).as_rotvec()
    axis_angles = axis_angles.reshape(num_frames, joints_num * 3)
    root_orient = torch.from_numpy(axis_angles[:, :3]).float().to(device)
    pose_body = torch.from_numpy(axis_angles[:, 3:66]).float().to(device)
    trans_t = torch.from_numpy(pos).float().to(device)
    betas = torch.zeros((num_frames, 10), device=device)
    if rep == '263':
        # Generate 263-dim HumanML3D features
        motion_feat = process_motion_smpl_263(
            root_orient=root_orient,
            pose_body=pose_body,
            betas=betas,
            trans=trans_t,
            mirror=mirror
        )
    elif rep == '272':
        # Generate 272-dim MotionStreamer features
        motion_feat = process_motion_smpl_272(
            root_orient=root_orient,
            pose_body=pose_body,
            betas=betas,
            trans=trans_t,
            mirror=mirror
        )
    else:
        raise ValueError(f"Unsupported representation: {rep}. Choose '263' or '272'.")

    return motion_feat


def process_motion_aistpp_151(motion_file, tgt_fps=30):
    """
    Process AIST++ motion into EDGE 151-dim format.
    Reference: https://github.com/Stanford-TML/EDGE
    aistpp is y-up
    
    Args:
        pos: root position, shape (seq_len, 3)
        q: local rotations in axis-angle, shape (seq_len, 24, 3)
    
    Returns:
        motion_151d: shape (seq_len, 151)
    
    """
    poses, scale, trans = load_motion_aistpp(motion_file)
    seq_len = pos.shape[0]
    if len(q.shape) == 2 and q.shape[1] == 72:
        q = q.reshape(seq_len, 24, 3)
    assert pos.shape == (seq_len, 3), f"Expected pos shape ({seq_len}, 3), got {pos.shape}"
    assert q.shape == (seq_len, 24, 3), f"Expected q shape ({seq_len}, 24, 3), got {q.shape}"
    
    pos = torch.Tensor(pos)
    q = torch.Tensor(q)
    
    # batch
    pos = pos.unsqueeze(0)  # (1, seq_len, 3)
    q = q.unsqueeze(0)  # (1, seq_len, 24, 3)
    
    
    # FK positions
    smpl = SMPLSkeleton()
    positions = smpl.forward(q, pos)  # (1, seq_len, 24, 3)
    
    # feet contact
    feet = positions[:, :, (7, 8, 10, 11)]  # (1, seq_len, 4, 3)
    feetv = torch.zeros(feet.shape[:3])  # (1, seq_len, 4)
    feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
    contacts = (feetv < 0.01).float()  # (1, seq_len, 4)
    

    # 6d
    q_6d = ax_to_6v(q)  # (1, seq_len, 24, 6)
    
    # contacts (4) + root_pos (3) + q_6d (24*6=144) = 151
    contacts_flat = contacts.reshape(1, seq_len, -1)  # (1, seq_len, 4)
    pos_flat = pos.reshape(1, seq_len, -1)  # (1, seq_len, 3)
    q_6d_flat = q_6d.reshape(1, seq_len, -1)  # (1, seq_len, 144)
    
    motion_151d = torch.cat([contacts_flat, pos_flat, q_6d_flat], dim=2)  # (1, seq_len, 151)
    
    motion_151d = motion_151d.squeeze(0)  # (seq_len, 151)
    
    return motion_151d.numpy()
    

########################## Calculate Dataset Statistics #####################
def mean_variance(data_dir, save_dir, rep='263'):
    """
    Calculate mean and variance for any representation given dims. 263 by default
    """
    joints_num = 22
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    if rep == '263':
        # 263-dim format: (r_a, r_x, r_z, r_y, jp, jv, jr, cf)
        # no root position. therefore local coordinates
        Std[0:1] = Std[0:1].mean() / 1.0                    # root angular velocity (r_a)
        Std[1:3] = Std[1:3].mean() / 1.0                    # root linear velocities (r_x, r_z)
        Std[3:4] = Std[3:4].mean() / 1.0                    # root height (r_y)
        Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0                                             # local joint positions (21 joints * 3)
        Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0   # local joint rotations in 6D (21 joints * 6)
        Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = \
            Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0                                      # local joint velocities (22 joints * 3)
        Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0         # foot contact features
        assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]                                                        # 4 + (21*3) + (21*6) + (22*3) + 4 = = 263

    elif rep == '272':
        # 272-dim format: (r_x, r_z, r_a, jp, jv, jr)
        # add root rotation in 6d, enabling ik-free rendering, thus better for autoregressive real-time tasks
        # no root position(in jp, root position should always be (0,0,0) because jp is wrt the root space). therefore local coordinates
        Std[:2] = Std[:2].mean() / 1.0                                                          # root linear velocities (r_x, r_z)
        Std[2:8] = Std[2:8].mean() / 1.0                                                        # root angular velocity in 6D (r_a)
        Std[8:8+3*joints_num] = Std[8:8+3*joints_num].mean() / 1.0                              # local joint positions (22 joints * 3)
        Std[8+3*joints_num:8+6*joints_num] = Std[8+3*joints_num:8+6*joints_num].mean() / 1.0    # local joint velocities (22 joints * 3)
        Std[8+6*joints_num:8+12*joints_num] = Std[8+6*joints_num:8+12*joints_num].mean() / 1.0  # local joint rotations in 6D (22 joints * 6)
        # Total: 2 + 6 + 66 + 66 + 132 = 272

    elif rep == '151':
        data_mean = data.mean(dim=0)  # (151,)  mean
        data_std = data.std(dim=0)    # (151,)  std
        # root position used. therefore global coordinates
        data_mean[0:4] = data_mean[0:4].mean()           # foot contacts
        data_mean[4:7] = data_mean[4:7].mean()           # root position
        data_mean[7:13] = data_mean[7:13].mean()         # root rotation (6d)
        data_mean[13:151] = data_mean[13:151].mean()     # local rotations (23 joints × 6d)
        
        data_std[0:4] = data_std[0:4].mean()           # contacts
        data_std[4:7] = data_std[4:7].mean()           # root position
        data_std[7:13] = data_std[7:13].mean()         # root rotation (6d)
        data_std[13:151] = data_std[13:151].mean()     # local rotations (23 joints × 6d)

    os.makedirs(save_dir, exist_ok=True)
    np.save(pjoin(save_dir, f'Mean_{int(rep)}.npy'), Mean)
    np.save(pjoin(save_dir, f'Std_{int(rep)}.npy'), Std)

    print("Mean and Std saved to", save_dir)

    return Mean, Std


def mean_variance_multi(data_dirs, save_dir, rep='272'):
    """
    Calculate mean and variance for multiple data directories combined.
    
    Args:
        data_dirs: List of data directory paths
        save_dir: Directory to save Mean and Std files
        rep: Representation type ('263', '272', or '151')
    """
    joints_num = 22
    data_list = []
    
    # Load data from all directories
    for data_dir in data_dirs:
        print(f"Loading data from {data_dir}...")
        file_list = os.listdir(data_dir)
        file_list = [f for f in file_list if f.endswith('.npy')]
        
        for file in file_list:
            data = np.load(pjoin(data_dir, file))
            if np.isnan(data).any():
                print(f"Skipping {file} (contains NaN)")
                continue
            data_list.append(data)
        
        print(f"  Loaded {len([f for f in file_list if f.endswith('.npy')])} files")
    
    # Concatenate all data
    print(f"\nConcatenating data from {len(data_list)} files...")
    data = np.concatenate(data_list, axis=0)
    print(f"Combined data shape: {data.shape}")
    
    # Calculate mean and std
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    
    # Normalize std based on representation type
    if rep == '263':
        # 263-dim format: (r_a, r_x, r_z, r_y, jp, jv, jr, cf)
        Std[0:1] = Std[0:1].mean() / 1.0                    # root angular velocity (r_a)
        Std[1:3] = Std[1:3].mean() / 1.0                    # root linear velocities (r_x, r_z)
        Std[3:4] = Std[3:4].mean() / 1.0                    # root height (r_y)
        Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
        Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
        Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = \
            Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
        Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0
        assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]
    
    elif rep == '272':
        # 272-dim format: (r_x, r_z, r_a, jp, jv, jr)
        Std[:2] = Std[:2].mean() / 1.0                                                          # root linear velocities (r_x, r_z)
        Std[2:8] = Std[2:8].mean() / 1.0                                                        # root angular velocity in 6D (r_a)
        Std[8:8+3*joints_num] = Std[8:8+3*joints_num].mean() / 1.0                              # local joint positions (22 joints * 3)
        Std[8+3*joints_num:8+6*joints_num] = Std[8+3*joints_num:8+6*joints_num].mean() / 1.0    # local joint velocities (22 joints * 3)
        Std[8+6*joints_num:8+12*joints_num] = Std[8+6*joints_num:8+12*joints_num].mean() / 1.0  # local joint rotations in 6D (22 joints * 6)
        # Total: 2 + 6 + 66 + 66 + 132 = 272
    
    elif rep == '151':
        Mean[0:4] = Mean[0:4].mean()           # foot contacts
        Mean[4:7] = Mean[4:7].mean()           # root position
        Mean[7:13] = Mean[7:13].mean()         # root rotation (6d)
        Mean[13:151] = Mean[13:151].mean()     # local rotations (23 joints × 6d)
        
        Std[0:4] = Std[0:4].mean()           # contacts
        Std[4:7] = Std[4:7].mean()           # root position
        Std[7:13] = Std[7:13].mean()         # root rotation (6d)
        Std[13:151] = Std[13:151].mean()     # local rotations (23 joints × 6d)
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    np.save(pjoin(save_dir, f'Mean_{int(rep)}.npy'), Mean)
    np.save(pjoin(save_dir, f'Std_{int(rep)}.npy'), Std)
    
    print(f"\nMean and Std saved to {save_dir}")
    print(f"  Mean_{int(rep)}.npy shape: {Mean.shape}")
    print(f"  Std_{int(rep)}.npy shape: {Std.shape}")
    
    return Mean, Std


########################## Process Dataset Function #######################
# Process entire AIST++ dataset
def process_dataset_aistpp(input_dir, output_dir, tgt_fps=30, rep='263', make_animation=False, mirror=False):
    """Process entire AIST++ dataset into specified representation and fps."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if make_animation:
        animation_dir = output_dir.replace("motions", "animations")
        if not os.path.exists(animation_dir):
            os.makedirs(animation_dir)
    motion_files = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".pkl")
    ]

    for motion_file in tqdm.tqdm(motion_files):
        input_path = os.path.join(input_dir, motion_file)
        output_path = os.path.join(output_dir, motion_file.replace(".pkl", ".npy"))

        if rep in ['263', '272']:
            motion_feat = process_motion_aistpp(input_path, tgt_fps=tgt_fps, rep=rep, mirror=mirror)
        elif rep == '151':
            motion_feat = process_motion_aistpp_151(input_path, tgt_fps=tgt_fps)
        else:
            raise ValueError(f"Unsupported representation: {rep}")     
        if make_animation:
            # Ensure we generate an mp4 animation path even though output_path is .npy
            animation_path = output_path.replace("motions", "animations").replace(".npy", ".mp4")
            music_path = input_path.replace("motions", "wavs").replace(".pkl", ".wav")
            # global_positions, fps -> save animation
            render_single_video(
                motion_feat,
                save_path=animation_path,
                dim=int(rep),
                render_setting={"fps": tgt_fps,
                                "music_path": music_path},
            )
        numpy.save(output_path, motion_feat)

def process_dataset_finedance(input_dir, output_dir, tgt_fps=30, rep='263', make_animation=False, mirror=False):
    """Process entire FineDance dataset into specified representation and fps."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if make_animation:
        animation_dir = output_dir.replace("motion", "animations")
        if not os.path.exists(animation_dir):
            os.makedirs(animation_dir)
    motion_files = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".npy")
    ]

    for motion_file in tqdm.tqdm(motion_files):
        input_path = os.path.join(input_dir, motion_file)
        output_path = os.path.join(output_dir, motion_file)

        if rep in ['263', '272']:
            motion_feat = process_motion_finedance(input_path, tgt_fps=tgt_fps, rep=rep, mirror=mirror)
        else:
            raise ValueError(f"Unsupported representation: {rep}")
        
        if make_animation:
            # Ensure we generate an mp4 animation path even though output_path is .npy
            animation_path = output_path.replace("motion", "animations").replace(".npy", ".mp4")
            music_path = input_path.replace("motion", "music_wav").replace(".npy", ".wav")
            # global_positions, fps -> save animation
            render_single_video(
                motion_feat,
                save_path=animation_path,
                dim=int(rep),
                render_setting={"fps": tgt_fps,
                                "music_path": music_path},
            )
        numpy.save(output_path, motion_feat)
        
        


########################## Animation-only Utility #######################
def _default_animations_dir_from_motion_dir(motion_dir: str) -> str:
    """
    Derive a default animations directory name from a processed motion directory.
    Examples:
      motions_272_30fps -> animations_272_30fps
      motion_272_30fps  -> animations_272_30fps
      <other>           -> <other>_animations
    """
    p = Path(motion_dir)
    name = p.name
    if "motions" in name:
        name = name.replace("motions", "animations")
    elif "motion" in name:
        name = name.replace("motion", "animations")
    else:
        name = name + "_animations"
    return str(p.parent / name)


def make_animation_from_processed_dir(
    motion_dir: str,
    out_dir: Optional[str] = None,
    dim: int = 272,
    fps: int = 30,
    dataset: Optional[str] = None,
    music_dir: Optional[str] = None,
    require_music: bool = False,
):
    """
    Render mp4 animations from *already processed* motion npy files (e.g., 272-dim).

    Args:
        motion_dir: directory containing processed .npy motions
        out_dir: directory to write .mp4 files; if None, inferred from motion_dir
        dim: motion representation dimension (typically 272)
        fps: fps used for rendering
        dataset: optional hint for auto music lookup: 'aistpp' or 'finedance'
        music_dir: optional explicit directory containing .wav files (overrides dataset)
        require_music: if True, raise when a matching wav is missing
    """
    motion_dir_p = Path(motion_dir)
    if not motion_dir_p.exists():
        raise FileNotFoundError(f"motion_dir not found: {motion_dir}")
    if not motion_dir_p.is_dir():
        raise NotADirectoryError(f"motion_dir is not a directory: {motion_dir}")

    if out_dir is None:
        out_dir = _default_animations_dir_from_motion_dir(motion_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Resolve music dir
    music_dir_p: Optional[Path]
    if music_dir is not None:
        music_dir_p = Path(music_dir)
    else:
        if dataset == "aistpp":
            music_dir_p = ROOT / "raw_data" / "aistpp" / "wavs"
        elif dataset == "finedance":
            music_dir_p = ROOT / "raw_data" / "finedance" / "music_wav"
        else:
            music_dir_p = None

    motion_files = sorted([p for p in motion_dir_p.iterdir() if p.is_file() and p.suffix == ".npy"])
    if len(motion_files) == 0:
        raise FileNotFoundError(f"No .npy files found under: {motion_dir}")

    for motion_path in tqdm.tqdm(motion_files):
        motion_feat = np.load(str(motion_path))
        save_path = out_dir_p / (motion_path.stem + ".mp4")

        render_setting = {"fps": fps}
        if music_dir_p is not None:
            wav_path = music_dir_p / (motion_path.stem + ".wav")
            if wav_path.exists():
                render_setting["music_path"] = str(wav_path)
            elif require_music:
                raise FileNotFoundError(f"Missing wav for {motion_path.name}: {wav_path}")

        render_single_video(
            motion_feat,
            save_path=str(save_path),
            dim=int(dim),
            render_setting=render_setting,
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Process motion dataset OR render animations from processed motions')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['process', 'animate'],
        default='process',
        help="process: convert raw dataset to npy; animate: render mp4 from processed npy",
    )
    
    parser.add_argument(
        '--dim', 
        type=str, 
        choices=['263', '272', '151'],
        default='263',
        help='Motion representation dimension: 263 (HumanML3D), 272 (MotionStreamer), or 151 (EDGE)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['finedance', 'aistpp'],
        default='aistpp',
        help='Dataset type: finedance or aistpp'
    )

    parser.add_argument(
        '--input_motion_dir',
        type=str,
        default=None,
        help='[animate mode] Directory containing processed motion .npy files (e.g., motions_272_30fps)',
    )

    parser.add_argument(
        '--out_animation_dir',
        type=str,
        default=None,
        help='[animate mode] Directory to write mp4 files (default: inferred from input_motion_dir)',
    )

    parser.add_argument(
        '--music_dir',
        type=str,
        default=None,
        help='[animate mode] Optional directory containing .wav files (overrides --dataset auto lookup)',
    )

    parser.add_argument(
        '--require_music',
        action='store_true',
        help='[animate mode] If set, raise error when matching wav is missing',
    )
    
    parser.add_argument(
        '--tgt_fps',
        type=int,
        default=30,
        help='Target frames per second for output motion'
    )
    
    parser.add_argument(
        '--make_animation',
        action='store_true',
        help='Whether to generate animation videos'
    )

    parser.add_argument(
        '--mirror',
        action='store_true',
        help='Whether to mirror the motion data'
    )
    
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    
    mode = args.mode
    rep = args.dim
    dataset = args.dataset
    tgt_fps = args.tgt_fps
    make_animation = args.make_animation
    mirror = args.mirror
    
    if mode == "animate":
        if args.input_motion_dir is None:
            raise ValueError("--input_motion_dir is required when --mode animate")
        make_animation_from_processed_dir(
            motion_dir=args.input_motion_dir,
            out_dir=args.out_animation_dir,
            dim=int(rep),
            fps=tgt_fps,
            dataset=dataset,
            music_dir=args.music_dir,
            require_music=bool(args.require_music),
        )
        print(f"Animation rendering completed! Output saved to {args.out_animation_dir or _default_animations_dir_from_motion_dir(args.input_motion_dir)}")
    else:
        print(f"Processing {dataset} dataset with {rep}-dim representation at {tgt_fps}fps")
        
        if dataset == 'finedance':
            data_directory = "raw_data/finedance"
            input_directory = "raw_data/finedance/motion"
            output_directory = f"raw_data/finedance/motion_{rep}_{tgt_fps}fps"
            if mirror:
                output_directory += "_mirror"
            
            process_dataset_finedance(
                input_directory, 
                output_directory, 
                tgt_fps=tgt_fps, 
                rep=rep, 
                make_animation=make_animation,
                mirror=mirror
            )            
        elif dataset == 'aistpp':
            data_directory = "raw_data/aistpp"
            input_directory = "raw_data/aistpp/motions"
            output_directory = f"raw_data/aistpp/motions_{rep}_{tgt_fps}fps"

            if mirror:
                output_directory += "_mirror"
            
            process_dataset_aistpp(
                input_directory, 
                output_directory, 
                tgt_fps=tgt_fps, 
                rep=rep, 
                make_animation=make_animation,
                mirror=mirror
            )

            mean, std = mean_variance(output_directory, data_directory, rep=rep)

        print(f"Processing completed! Output saved to {output_directory}")
        # Examples:
        # python utils/dataset_process.py --mode process --dim 272 --dataset aistpp --make_animation --tgt_fps 30
        # python utils/dataset_process.py --mode animate --dim 272 --dataset finedance --input_motion_dir raw_data/finedance/motion_272_30fps --music_dir raw_data/finedance/music_wav --tgt_fps 30