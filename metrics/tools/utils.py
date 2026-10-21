# This module is adapted from text-to-motion evaluation code
# Original source: https://github.com/EricGuo5513/text-to-motion
# Used for text-to-motion evaluation metrics
# License: MIT

import sys

import numpy as np
import scipy.linalg
import torch
from torch import linalg
from scipy.ndimage import uniform_filter1d


def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)


def variance(x, T, dim):
    mean = x.mean(dim)
    out = (x - mean) ** 2
    out = out.sum(dim)
    return out / (T - 1)


def sqrtm(input):
    m = input.detach().cpu().numpy().astype(np.float64_)
    sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m)).to(input)
    return sqrtm


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * torch.mm(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = torch.sum(torch.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = torch.sum(torch.square(matrix2), axis=1)  # shape (num_train, )
    dists = torch.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def euclidean_distance_matrix_np(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = (
        torch.unsqueeze(torch.arange(size), 1).to(mat.device).repeat_interleave(size, 1)
    )
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = correct_vec | bool_mat[:, i]
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = torch.cat(top_k_list, dim=1)
    return top_k_mat


def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_activation_statistics_np(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, (
        "Training and test mean vectors have different lengths"
    )
    assert sigma1.shape == sigma2.shape, (
        "Training and test covariances have different dimensions"
    )

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        #     m = np.max(np.abs(covmean.imag))
        #     raise ValueError("Imaginary component {}".format(m))
            # print("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

def calculate_avg_distance(feature_list, mean=None, std=None):
    assert len(feature_list.shape) == 2
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist


def calculate_diversity_np(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = scipy.linalg.norm(
        activation[first_indices] - activation[second_indices], axis=1
    )
    return dist.mean()


def calculate_multimodality_np(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = scipy.linalg.norm(
        activation[:, first_dices] - activation[:, second_dices], axis=2
    )
    return dist.mean()


# motion reconstructions metrics


def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat, (scale, R, t)


def compute_mpjpe(preds, target, valid_mask=None, pck_joints=None, sample_wise=True):
    """
    Mean per-joint position error (i.e. mean Euclidean distance)
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape, print(preds.shape, target.shape)  # BxJx3
    mpjpe = torch.norm(preds - target, p=2, dim=-1)  # BxJ

    if pck_joints is None:
        if sample_wise:
            mpjpe_seq = (
                (mpjpe * valid_mask.float()).sum(-1) / valid_mask.float().sum(-1)
                if valid_mask is not None
                else mpjpe.mean(-1)
            )
        else:
            mpjpe_seq = mpjpe[valid_mask] if valid_mask is not None else mpjpe
        return mpjpe_seq
    else:
        mpjpe_pck_seq = mpjpe[:, pck_joints]
        return mpjpe_pck_seq


def align_by_parts(joints, align_inds=None):
    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1)
    return joints - torch.unsqueeze(pelvis, dim=1)


def calc_mpjpe(preds, target, align_inds=[0], sample_wise=True, trans=None):
    # Expects BxJx3
    valid_mask = target[:, :, 0] != -2.0
    # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
    if align_inds is not None:
        preds_aligned = align_by_parts(preds, align_inds=align_inds)
        if trans is not None:
            preds_aligned += trans
        target_aligned = align_by_parts(target, align_inds=align_inds)
    else:
        preds_aligned, target_aligned = preds, target
    mpjpe_each = compute_mpjpe(
        preds_aligned, target_aligned, valid_mask=valid_mask, sample_wise=sample_wise
    )
    return mpjpe_each


def calc_accel(preds, target):
    """
    Mean joint acceleration error
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape, print(preds.shape, target.shape)  # BxJx3
    assert preds.dim() == 3
    # Expects BxJx3
    # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
    accel_gt = target[:-2] - 2 * target[1:-1] + target[2:]
    accel_pred = preds[:-2] - 2 * preds[1:-1] + preds[2:]
    normed = torch.linalg.norm(accel_pred - accel_gt, dim=-1)
    accel_seq = normed.mean(1)
    return accel_seq


def calc_pampjpe(preds, target, sample_wise=True, return_transform_mat=False):
    # Expects BxJx3
    target, preds = target.float(), preds.float()
    # extracting the keypoints that all samples have valid annotations
    # valid_mask = (target[:, :, 0] != -2.).sum(0) == len(target)
    # preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(preds[:, valid_mask], target[:, valid_mask])
    # pa_mpjpe_each = compute_mpjpe(preds_tranformed, target[:, valid_mask], sample_wise=sample_wise)

    preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(
        preds, target
    )
    pa_mpjpe_each = compute_mpjpe(preds_tranformed, target, sample_wise=sample_wise)

    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each


# from action2motion
def calculate_diversity_multimodality(
    activations, labels, num_labels, diversity_times=200, multimodality_times=20
):
    labels = labels.long()
    num_motions = activations.shape[0]  # len(labels)

    diversity = 0

    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices, strict=False):
        diversity += torch.dist(activations[first_idx, :], activations[second_idx, :])
    diversity /= diversity_times

    multimodality = 0
    label_quotas = np.zeros(num_labels)
    label_quotas[labels.unique()] = (
        multimodality_times  # if a label does not appear in batch, its quota remains zero
    )
    while np.any(label_quotas > 0):
        # print(label_quotas)
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx]
        if not label_quotas[first_label]:
            continue

        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx]
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]

        label_quotas[first_label] -= 1

        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation, second_activation)

    multimodality /= multimodality_times * num_labels

    return diversity, multimodality


def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance_np(
        statistics_1[0], statistics_1[1], statistics_2[0], statistics_2[1]
    )


# from: https://github.com/abdulfatir/gan-metrics-pytorch/blob/master/kid_score.py
def polynomial_mmd_averages(
    codes_g,
    codes_r,
    n_subsets=50,
    subset_size=1000,
    ret_var=True,
    output=sys.stdout,
    **kernel_args,
):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    replace = subset_size < len(codes_g)

    for i in range(n_subsets):
        g = codes_g[choice(len(codes_g), subset_size, replace=replace)]
        r = codes_r[choice(len(codes_r), subset_size, replace=replace)]
        o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
        if ret_var:
            mmds[i], vars[i] = o
        else:
            mmds[i] = o

    return (mmds, vars) if ret_var else mmds


def polynomial_mmd(
    codes_g, codes_r, degree=3, gamma=None, coef0=1, var_at_m=None, ret_var=True
):
    from sklearn.metrics.pairwise import polynomial_kernel

    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY, var_at_m=var_at_m, ret_var=ret_var)


def _mmd2_and_variance(
    K_XX,
    K_XY,
    K_YY,
    unit_diagonal=False,
    mmd_est="unbiased",
    block_size=1024,
    var_at_m=None,
    ret_var=True,
):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == "biased":
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2 * K_XY_sum / (m * m)
        )
    else:
        assert mmd_est in {"unbiased", "u-statistic"}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
        if mmd_est == "unbiased":
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1
        / (m * m1 * m2)
        * (_sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (_sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (
        4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
        + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est
    )

    return mmd2, var_est


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def calculate_kid(real_activations, generated_activations):
    kid_values = polynomial_mmd_averages(
        real_activations, generated_activations, n_subsets=100
    )
    results = (kid_values[0].mean(), kid_values[0].std())
    return results

#####################Physics Metrics################################
def calculate_skating_ratio(motions):
    '''
    Code adopted from the GMD codebase.
    '''
    thresh_height = 0.05 # 10
    fps = 20.0
    thresh_vel = 0.50 # 20 cm /s
    avg_window = 5 # frames

    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # [bs, max_len -1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]

    return skating_ratio, skate_vel


def calculate_mean_penetration(motions, mask=None):
    '''
    Mean penetration of the joints into the ground.
    '''
    import torch
    # samples shape:  B x n_joints x 3 x L
    joint_heights = motions[:, :, 1]
    penetration = torch.clamp(joint_heights, max=0) # B x n_joints x L
    if mask is not None:
        penetration *= mask.view(mask.shape[0], 1, mask.shape[-1], 1)
    per_joint_penalty = penetration.amin(dim=1).sum() * -1.
    per_joint_frame_penalty = penetration.amin(dim=1).amin(dim=-1).sum() * -1.
    non_zero_count = torch.sum(penetration != 0, dtype=torch.float32)

    return per_joint_frame_penalty.numpy() * 100. #/ non_zero_count # convert to centimeters


def calculate_penetration(motions, lengths):
    '''
    Based on PhysDiff's implementation.
    '''
    penetration_list = []
    joint_heights = motions[:, :, 1, :] # B x n_joints x L 
    lowest_heights = joint_heights.min(dim=1)[0] # B x L
    tolerance = 0.005
    for i, motion_len in enumerate(lengths):
        penetration = (lowest_heights[i, :motion_len] + tolerance).clamp_max(0) * 1000 * -1.
        penetration_list += penetration.tolist()
    
    return np.mean(penetration_list)


def calculate_floating(motions, lengths):
    '''
    Based on PhysDiff's implementation.
    '''
    floating_list = []
    joint_heights = motions[:, :, 1, :] # B x n_joints x L
    lowest_heights = joint_heights.min(dim=1)[0] # B x L
    tolerance = 0.005
    for i, motion_len in enumerate(lengths):
        floating = (lowest_heights[i, :motion_len] - tolerance).clamp_min(0) * 1000
        floating_list += floating.tolist()

    return np.mean(floating_list)


def calculate_foot_sliding(motions, lengths):
    '''
    Based on PhysDiff's implementation.
    '''
    import torch
    skating_list = []
    margin = 0.005
    for joints, motion_len in zip(motions, lengths):
        # joints: n_joints x 3 x L
        for t in range(motion_len - 1):
            contact_idx = joints[:, 1, t].min(dim=-1)[1]
            if joints[contact_idx, 1, t] <= margin and joints[contact_idx, 1, t+1] <= margin:
                offset = joints[contact_idx, [0, 2], t+1] - joints[contact_idx, [0, 2], t] # horizontal distance
                skate_i = torch.norm(offset).mean().item() * 1000
            else:
                skate_i = 0.
            skating_list.append(skate_i)
    
    return np.mean(skating_list)

def calculate_foot_skating_ratio(joints: np.ndarray, ground_height=0.0, ankle_threshold=0.08, toe_threshold=0.05):
    """
    Calculate foot skating ratio for a single motion sequence.
    Args:
        joints: Joint positions with shape (T, 22, 3)
    """
    l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
    joints = torch.from_numpy(joints).float()  # (S, 22, 3)
    relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]
    pred_joint_xyz = joints[:, relevant_joints, :] 
    pred_vel = torch.zeros_like(pred_joint_xyz)
    pred_vel[:-1] = (
        pred_joint_xyz[1:, :, :] - pred_joint_xyz[:-1, :, :]
    )  # (S-1, 4, 3)
    left_foot_y_ankle = joints[ :, l_ankle_idx, 1]
    right_foot_y_ankle = joints[ :, r_ankle_idx, 1]
    left_foot_y_toe = joints[:, l_foot_idx, 1]
    right_foot_y_toe = joints[:, r_foot_idx, 1]

    left_fc_mask = (left_foot_y_ankle <= (ground_height + ankle_threshold)) & (left_foot_y_toe <= (ground_height + toe_threshold))
    right_fc_mask = (right_foot_y_ankle <= (ground_height + ankle_threshold)) & (right_foot_y_toe <= (ground_height + toe_threshold))
    # fc_mask_y = torch.cat([fc_mask_ankle, fc_mask_teo], dim=1).squeeze(0)
    left_pred_vel = torch.cat([pred_vel[:, 0:1, :], pred_vel[:, 2:3, :]], dim=1)
    right_pred_vel = torch.cat([pred_vel[:, 1:2, :], pred_vel[:, 3:4, :]], dim=1)
    left_pred_vel[~left_fc_mask] = 0
    right_pred_vel[~right_fc_mask] = 0
    left_static_num = torch.sum(left_fc_mask)
    right_static_num = torch.sum(right_fc_mask)
    left_velocity_foot_tangent = torch.cat([left_pred_vel[:, :, 0:1], left_pred_vel[:, :, 2:3] ], dim=2)
    left_velocity_foot_tangent = torch.abs(torch.mean(left_velocity_foot_tangent, dim=-1))        # T, 4
    right_velocity_foot_tangent = torch.cat([right_pred_vel[:, :, 0:1], right_pred_vel[:, :, 2:3] ], dim=2)
    right_velocity_foot_tangent = torch.abs(torch.mean(right_velocity_foot_tangent, dim=-1))        # T, 4

    left_velocity_foot_tangent = left_velocity_foot_tangent > 0.01 #(0.05 / 30)          # 0.025     # 0.1/30
    right_velocity_foot_tangent = right_velocity_foot_tangent > 0.01 #(0.05 / 30)
    left_slide_frames = torch.any(left_velocity_foot_tangent, dim=-1)
    left_slide_num = torch.sum(left_slide_frames)
    left_ratio = left_slide_num / (left_static_num + 1e-9)
    left_ratio = left_ratio.item()

    right_slide_frames = torch.any(right_velocity_foot_tangent, dim=-1)
    right_slide_num = torch.sum(right_slide_frames)
    right_ratio = right_slide_num / (right_static_num + 1e-9)
    right_ratio = right_ratio.item()
    skating_ratio = (left_ratio + right_ratio) / 2.0
    return skating_ratio

def calculate_foot_skating_ratio_single(
    joint_positions: np.ndarray,
    seq_len: int,
    fps: float = 60.0,
    thresh_height: float = 0.05,
    thresh_vel: float = 0.50,
    avg_window: int = 5,
) -> float:
    """Calculate foot skating ratio for a single motion sequence.
    
    Args:
        joint_positions: Joint positions with shape (T, 22, 3), where
                        - T is the number of frames
                        - 22 is the number of joints
                        - 3 is (x, y, z) coordinates (y is up)
        seq_len: Valid sequence length
        fps: Frame rate (default 30.0)
        thresh_height: Height threshold for foot contact (default 0.05m)
        thresh_vel: Velocity threshold for skating detection (default 0.50 m/s)
        avg_window: Window size for velocity averaging (default 5)
    
    Returns:
        Skating ratio (0.0 to 1.0)
    
    Note:
        Foot joints are at indices 10 (left foot) and 11 (right foot).
        Skating is detected when:
        1. Foot is in contact with ground (height < thresh_height)
        2. Foot has horizontal velocity > thresh_vel
    """
    # Truncate to valid length
    joints = joint_positions[:seq_len]  # (seq_len, 22, 3)
    
    if seq_len <= 1:
        return 0.0
    
    # Extract foot joints: indices 10 (left) and 11 (right)
    # joints shape: (T, 22, 3) -> feet shape: (T, 2, 3)
    feet = joints[:, [10, 11], :]  # (T, 2, 3)
    
    # Calculate horizontal velocity (XZ plane)
    # feet[:, :, [0, 2]] extracts x and z coordinates
    feet_xz = feet[:, :, [0, 2]]  # (T, 2, 2)
    feet_vel = np.linalg.norm(feet_xz[1:] - feet_xz[:-1], axis=2) * fps  # (T-1, 2)
    
    # Apply moving average to velocity
    vel_avg = uniform_filter1d(feet_vel, axis=0, size=avg_window, mode='constant', origin=0)
    
    # Get foot heights (y coordinate)
    feet_height = feet[:, :, 1]  # (T, 2)
    
    # Detect foot contact: both current and next frame below threshold
    feet_contact = np.logical_and(
        feet_height[:-1] < thresh_height,
        feet_height[1:] < thresh_height
    )  # (T-1, 2)
    
    # Detect skating: foot is in contact AND moving fast
    skating = np.logical_and(feet_contact, feet_vel > thresh_vel)
    skating = np.logical_and(skating, vel_avg > thresh_vel)
    
    # Either foot skating counts
    skating_any = np.logical_or(skating[:, 0], skating[:, 1])  # (T-1,)
    
    # Calculate ratio
    skating_ratio = np.sum(skating_any) / len(skating_any)
    
    return float(skating_ratio)


##########################Music Metrics######################


def motion_peak_onehot(joints: np.ndarray, order: int = 10) -> np.ndarray:
    """Calculate motion beats from joint positions.
    
    Args:
        joints: Joint positions with shape (nframes, njoints, 3)
        order: Window size for local minima detection (default 10 for 60FPS, use 5 for 30FPS)
    
    Returns:
        peak_onehot: Boolean array of shape (nframes,) indicating motion beats
    """
    import scipy.signal as scisignal
    
    # Calculate velocity
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=order)
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1

    return peak_onehot


def beat_alignment_score(
    music_beats: np.ndarray, 
    motion_beats: np.ndarray, 
    sigma: float = 3.0
) -> float:
    """Calculate alignment score between music and motion beats.
    
    Uses a Gaussian kernel to measure how well motion beats align with music beats.
    
    Args:
        music_beats: Boolean/binary array indicating music beat positions
        motion_beats: Boolean/binary array indicating motion beat positions
        sigma: Standard deviation for Gaussian scoring (default 3.0)
    
    Returns:
        Alignment score between 0 and 1
    """
    if motion_beats.sum() == 0:
        return 0.0
    
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]
    
    if len(music_beat_idxs) == 0:
        return 0.0
    
    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        ind = np.argmin(dists)
        score = np.exp(-dists[ind] ** 2 / 2 / sigma ** 2)
        score_all.append(score)
    
    return sum(score_all) / len(score_all)


def extract_audio_beat_from_features(audio_features: np.ndarray) -> np.ndarray:
    """Extract beat onehot from baseline audio features.
    
    The baseline features have shape (frames, 35) where:
    - envelope[:, 0]: onset envelope
    - mfcc[:, 1:21]: 20 MFCC coefficients
    - chroma[:, 21:33]: 12 chroma features
    - peak_onehot[:, 33]: onset peaks
    - beat_onehot[:, 34]: beat onehot (last column)
    
    Args:
        audio_features: Audio features array of shape (frames, 35)
    
    Returns:
        beat_onehot: Boolean array of shape (frames,) indicating beat positions
    """
    return audio_features[:, -1].astype(bool)