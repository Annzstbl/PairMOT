'''这是一个实验性文件，还没有验证'''



from typing import Tuple

import torch
from torch import Tensor
from torch.autograd import Function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

EPSILON = 1e-8

eps_parallel = 1e-12
tol_seg = 1e-9
eps_abs = 1e-9
eps_rel = 1e-9
tol_len=1e-6

def deduplicate_vertices_mask(
    vertices: Tensor,  # (B,N,K,2)
    mask: Tensor,      # (B,N,K)
) -> Tensor:
    """
    Returns:
      new_mask: (B,N,K) where near-duplicate points are suppressed (keep smallest index)
    """
    v = vertices.double()
    m = mask.bool()

    # pairwise squared distance
    diff = v.unsqueeze(3) - v.unsqueeze(2)            # (B,N,K,K,2)
    dist2 = (diff * diff).sum(dim=-1)                 # (B,N,K,K)
    same = dist2 <= (tol_len * tol_len)

    # only compare among valid points
    same = same & m.unsqueeze(3) & m.unsqueeze(2)

    # suppress duplicates: if i<j and same, drop j
    upper = torch.triu(same, diagonal=1)              # keep only i<j
    drop = upper.any(dim=2)                           # (B,N,K): whether j has any i<j same
    new_mask = m & (~drop)
    return new_mask

def box_intersection(corners1: Tensor,
                     corners2: Tensor) -> Tuple[Tensor, Tensor]:
    """Find intersection points of rectangles.
    Convention: if two edges are collinear, there is no intersection point.

    Args:
        corners1 (Tensor): (B, N, 4, 2) First batch of boxes.
        corners2 (Tensor): (B, N, 4, 2) Second batch of boxes.

    Returns:
        Tuple:
         - Tensor: (B, N, 4, 4, 2) Intersections.
         - Tensor: (B, N, 4, 4) Valid intersections mask.
    """
    # build edges from corners
    # B, N, 4, 4: Batch, Box, edge, point
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3)
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    # duplicate data to pair each edges from the boxes
    # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(3)
    line2_ext = line2.unsqueeze(2)
    x1, y1, x2, y2 = line1_ext.split([1, 1, 1, 1], dim=-1)
    x3, y3, x4, y4 = line2_ext.split([1, 1, 1, 1], dim=-1)
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    numerator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)


    # parallel / nearly-parallel lines
    parallel = numerator.abs() < eps_parallel

    denumerator_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    denumerator_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)

    # compute t,u only where not parallel to avoid huge division noise
    t = torch.zeros_like(denumerator_t)
    u = torch.zeros_like(denumerator_u)
    valid = ~parallel
    t[valid] = denumerator_t[valid] / numerator[valid]
    u[valid] = -denumerator_u[valid] / numerator[valid]
    mask_t = (t >= -tol_seg) & (t <= 1 + tol_seg)  # intersection on line segment 1
    mask_u = (u >= -tol_seg) & (u <= 1 + tol_seg)  # intersection on line segment 2
    mask = mask_t & mask_u & ~parallel

    # intersection points (only meaningful for mask=True)
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    intersections = torch.stack([ix, iy], dim=-1) 
    intersections = intersections.squeeze(-2)      

    mask2 = mask.squeeze(-1)                       
    intersections = intersections * mask2.to(intersections.dtype).unsqueeze(-1)
    
    return intersections, mask


def box1_in_box2(corners1: Tensor, corners2: Tensor) -> Tensor:
    """Check if corners of box1 lie in box2.
    Convention: if a corner is exactly on the edge of the other box,
    it's also a valid point.

    Args:
        corners1 (Tensor): (B, N, 4, 2) First batch of boxes.
        corners2 (Tensor): (B, N, 4, 2) Second batch of boxes.

    Returns:
        Tensor: (B, N, 4) Intersection.
    """
    # a, b, c, d - 4 vertices of box2
    a = corners2[:, :, 0:1, :]  # (B, N, 1, 2)
    b = corners2[:, :, 1:2, :]  # (B, N, 1, 2)
    d = corners2[:, :, 3:4, :]  # (B, N, 1, 2)
    # ab, am, ad - vectors between corresponding vertices
    ab = b - a  # (B, N, 1, 2)
    am = corners1 - a  # (B, N, 4, 2)
    ad = d - a  # (B, N, 1, 2)
    prod_ab = torch.sum(ab * am, dim=-1)  # (B, N, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)  # (B, N, 1)
    prod_ad = torch.sum(ad * am, dim=-1)  # (B, N, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)  # (B, N, 1)

    tol_ab = eps_abs + eps_rel * norm_ab
    tol_ad = eps_abs + eps_rel * norm_ad

    cond1 = (prod_ab >= -tol_ab) & (prod_ab <= norm_ab + tol_ab)
    cond2 = (prod_ad >= -tol_ad) & (prod_ad <= norm_ad + tol_ad)

    # cond1 = (prod_ab >= 0) & (prod_ab <= norm_ab )   # (B, N, 4)
    # cond2 = (prod_ad >= 0) & (prod_ad <= norm_ad )   # (B, N, 4)
    return cond1 * cond2


def box_in_box(corners1: Tensor, corners2: Tensor) -> Tuple[Tensor, Tensor]:
    """Check if corners of two boxes lie in each other.

    Args:
        corners1 (Tensor): (B, N, 4, 2) First batch of boxes.
        corners2 (Tensor): (B, N, 4, 2) Second batch of boxes.

    Returns:
        Tuple:
         - Tensor: (B, N, 4) True if i-th corner of box1 is in box2.
         - Tensor: (B, N, 4) True if i-th corner of box2 is in box1.
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1


def build_vertices(corners1: Tensor, corners2: Tensor, c1_in_2: Tensor,
                   c2_in_1: Tensor, intersections: Tensor,
                   valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Find vertices of intersection area.

    Args:
        corners1 (Tensor): (B, N, 4, 2) First batch of boxes.
        corners2 (Tensor): (B, N, 4, 2) Second batch of boxes.
        c1_in_2 (Tensor): (B, N, 4) True if i-th corner of box1 is in box2.
        c2_in_1 (Tensor): (B, N, 4) True if i-th corner of box2 is in box1.
        intersections (Tensor): (B, N, 4, 4, 2) Intersections.
        valid_mask (Tensor): (B, N, 4, 4) Valid intersections mask.

    Returns:
        Tuple:
         - Tensor: (B, N, 24, 2) Vertices of intersection area;
               only some elements are valid.
         - Tensor: (B, N, 24) Mask of valid elements in vertices.
    """
    # NOTE: inter has elements equals zero and has zeros gradient
    # (masked by multiplying with 0); can be used as trick
    B = corners1.size()[0]
    N = corners1.size()[1]
    # (B, N, 4 + 4 + 16, 2)
    vertices = torch.cat(
        [corners1, corners2,
         intersections.view([B, N, -1, 2])], dim=2)
    # Bool (B, N, 4 + 4 + 16)
    mask = torch.cat([c1_in_2, c2_in_1, valid_mask.view([B, N, -1])], dim=2)
    return vertices, mask



def sort_vertices(vertices:Tensor, mask:Tensor):
    """

    Args:
        vertices (Tensor): float (B, N, 24, 2)
        mask (Tensor): bool (B, N, 24)

    Returns:
        sorted vertices: (B, N, 9, 2)
    
    Note:
        why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X) 
        and X indicates the index of arbitary elements in the last 16 (intersections not corners) with 
        value 0 and mask False. (cause they have zero value and zero gradient)
    """
    # 先做近似去重（强烈建议）
    mask = deduplicate_vertices_mask(vertices, mask)

    num_valid = torch.sum(mask.int(), dim=2).int()      # (B, N)
    mean = torch.sum(vertices * mask.type(vertices.dtype).unsqueeze(-1), dim=2, keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
    vertices_normalized = vertices - mean       # normalization makes sorting easier
    idx_sorted = sort_vertice_th(vertices_normalized, mask, num_valid.long()).long()
    
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1,1,1,2])
    selected = torch.gather(vertices, 2, idx_ext)
    
    # zero padding for invalid vertices
    m = _generate_mask(selected.size(-2), num_valid+1, dtype=vertices.dtype, device=vertices.device)
    selected = selected * m.unsqueeze(-1)
    
    # set zero of too few vertices
    m = num_valid >= 3
    selected = selected * m.type(selected.dtype).unsqueeze(-1).unsqueeze(-1)
    
    return selected
    


def calculate_area(selected):
    """calculate area of intersection

    Args:
        idx_sorted (Tensor): (B, N, 9)
        vertices (Tensor): (B, N, 24, 2)
    
    return:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding 
    """
    total = selected[:, :, 0:-1, 0]*selected[:, :, 1:, 1] - selected[:, :, 0:-1, 1]*selected[:, :, 1:, 0]
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected


def oriented_box_intersection_2d(corners1:Tensor, corners2:Tensor):
    """calculate intersection area of 2d rectangles 

    Args:
        corners1 (Tensor): (B, N, 4, 2)
        corners2 (Tensor): (B, N, 4, 2)

    Returns:
        Tuple:
         - Tensor (B, N): Area of intersection.
         - Tensor (B, N, 9, 2): Vertices of polygon with zero padding.
    """
    intersections, valid_mask = box_intersection(corners1, corners2)
    c12, c21 = box_in_box(corners1, corners2)
    c12, c21 = check_overlap(corners1, corners2, c12, c21)
    vertices, mask = build_vertices(corners1, corners2, c12, c21,
                                    intersections, valid_mask)

    vertices_gathered = sort_vertices(vertices, mask)

    return calculate_area(vertices_gathered)

@torch.no_grad()
def check_overlap(corners1:Tensor, corners2:Tensor, cond12:Tensor, cond21:Tensor):
    """check if corners are overlapped and update the conditions. 
    useful to avoid incorrect intersection calculation. 
    Without this check, the intersection would have duplicated vertices, which makes the 
    shoelace-formula broken. 

    Args:
        corners1 (Tensor): (B, N, 4, 2)
        corners2 (Tensor): (B, N, 4, 2)
        cond12 (Tensor): bool, (B, N, 4)
        cond21 (Tensor): bool, (B, N, 4)

    Returns:
        Tensor: bool, (B, N, 4)
        Tensor: bool, (B, N, 4)
    """
    c_roll = corners2
    cd_roll = cond21
    for _ in range(4):
        c_roll = torch.roll(c_roll, shifts=1, dims=2)
        cd_roll = torch.roll(cd_roll, shifts=1, dims=2)
        crit = torch.all(corners1 == c_roll, dim=-1)
        cond12[crit] = True
        cd_roll[crit] = False
    return cond12, cd_roll

@torch.no_grad()
def sort_vertice_th(vertices_normalized:Tensor, mask:Tensor, num_valid:Tensor):
    """_summary_

    Args:
        vertices_normalized (Tensor): (B, N, 24, 2)
        mask (Tensor): (B, N, 24)
        num_valid (Tensor): (B, N)

    Returns:
        Tensor: (B, N, 9)
    """
    x = vertices_normalized[..., 0]
    y = vertices_normalized[..., 1]
    
    # sorting
    x[~mask] = -1e6
    y[~mask] = 1e-6
    ang = torch.atan2(y, x)
    index = torch.argsort(ang, dim=-1)  # (B, N, 24)
    
    # duplicate the first
    temp = index[..., :1].clone()   # (B, N, 1)
    index.scatter_(dim=-1, index=num_valid.unsqueeze(-1), src=temp.expand(-1, -1, index.size(-1)))
    return index[..., :9]


@torch.no_grad()
def _generate_mask(num: int, valid_num: Tensor, dtype, device):
    B, N = valid_num.size()
    ar = torch.arange(num, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(B, N, 1).to(device)
    mask = ar < valid_num.unsqueeze(-1)
    # NOTE: this expression doesn't work some earlier PyTorch version:
    # arr = torch.where(mask, 1., 0.)
    ar = torch.where(mask, torch.ones((1,)).expand_as(mask).to(device), torch.zeros(1,).expand_as(mask).to(device))
    return ar



def box2corners(box: Tensor) -> Tensor:
    """Convert rotated 2d box coordinate to corners.

    Args:
        box (Tensor): (B, N, 5) with x, y, w, h, alpha.

    Returns:
        Tensor: (B, N, 4, 2) Corners.
    """
    B = box.size()[0]
    x, y, w, h, alpha = box.split([1, 1, 1, 1, 1], dim=-1)
    x4 = box.new_tensor([0.5, -0.5, -0.5, 0.5]).to(box.device)
    x4 = x4 * w  # (B, N, 4)
    y4 = box.new_tensor([0.5, 0.5, -0.5, -0.5]).to(box.device)
    y4 = y4 * h  # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)  # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)  # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1, 4, 2]), rot_T.view([-1, 2, 2]))
    rotated = rotated.view([B, -1, 4, 2])  # (B * N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated


def diff_iou_rotated_2d(box1: Tensor, box2: Tensor) -> Tensor:
    iou, corners1, corners2, u = cal_iou(box1, box2)
    return iou

def cal_iou(box1:Tensor, box2:Tensor):
    """Calculate differentiable iou of rotated 2d boxes.

    Args:
        box1 (Tensor): (B, N, 5) First box.
        box2 (Tensor): (B, N, 5) Second box.

    Returns:
        Tensor: (B, N) IoU.
    """
    corners1 = box2corners(box1)
    corners2 = box2corners(box2)
    intersection, _ = oriented_box_intersection_2d(corners1,
                                                   corners2)  # (B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou, corners1, corners2,  union


def diff_diou_rotated_2d(box1:Tensor, box2:Tensor, enclosing_type:str="smallest"):
    """calculate diou loss

    Args:
        box1 (Tensor): [description]
        box2 (Tensor): [description]
    """
    iou, corners1, corners2, u = cal_iou(box1, box2)
    w, h = enclosing_box(corners1, corners2, enclosing_type)
    c2 = w*w + h*h      # (B, N)
    x_offset = box1[...,0] - box2[..., 0]
    y_offset = box1[...,1] - box2[..., 1]
    d2 = x_offset*x_offset + y_offset*y_offset
    # diou_loss = 1. - iou + d2/c2
    diou = iou - d2/c2
    return diou


def diff_giou_rotated_2d(box1:Tensor, box2:Tensor, enclosing_type:str="smallest"):
    iou, corners1, corners2, u = cal_iou(box1, box2)
    w, h = enclosing_box(corners1, corners2, enclosing_type)
    area_c =  w*h
    # giou_loss = 1. - iou + ( area_c - u )/area_c
    giou = iou - ( area_c - u )/area_c
    return giou


    
def enclosing_box(corners1:Tensor, corners2:Tensor, enclosing_type:str="smallest"):
    if enclosing_type == "aligned":
        return enclosing_box_aligned(corners1, corners2)
    elif enclosing_type == "pca":
        return enclosing_box_pca(corners1, corners2)
    elif enclosing_type == "smallest":
        return smallest_bounding_box(torch.cat([corners1, corners2], dim=-2))
    else:
        ValueError("Unknow type enclosing. Supported: aligned, pca, smallest")


def enclosing_box_aligned(corners1:Tensor, corners2:Tensor):
    """calculate the smallest enclosing box (axis-aligned)

    Args:
        corners1 (Tensor): (B, N, 4, 2)
        corners2 (Tensor): (B, N, 4, 2)
    
    Returns:
        w (Tensor): (B, N)
        h (Tensor): (B, N)
    """
    x1_max = torch.max(corners1[..., 0], dim=2)[0]     # (B, N)
    x1_min = torch.min(corners1[..., 0], dim=2)[0]     # (B, N)
    y1_max = torch.max(corners1[..., 1], dim=2)[0]
    y1_min = torch.min(corners1[..., 1], dim=2)[0]
    
    x2_max = torch.max(corners2[..., 0], dim=2)[0]     # (B, N)
    x2_min = torch.min(corners2[..., 0], dim=2)[0]    # (B, N)
    y2_max = torch.max(corners2[..., 1], dim=2)[0]
    y2_min = torch.min(corners2[..., 1], dim=2)[0]

    x_max = torch.max(x1_max, x2_max)
    x_min = torch.min(x1_min, x2_min)
    y_max = torch.max(y1_max, y2_max)
    y_min = torch.min(y1_min, y2_min)

    w = x_max - x_min       # (B, N)
    h = y_max - y_min
    return w, h


def enclosing_box_pca(corners1:Tensor, corners2:Tensor):
    """calculate the rotated smallest enclosing box using PCA

    Args:
        corners1 (Tensor): (B, N, 4, 2)
        corners2 (Tensor): (B, N, 4, 2)
    
    Returns:
        w (Tensor): (B, N)
        h (Tensor): (B, N)
    """
    B = corners1.size()[0]
    c = torch.cat([corners1, corners2], dim=2)      # (B, N, 8, 2)
    c = c - torch.mean(c, dim=2, keepdim=True)      # normalization
    c = c.view([-1, 8, 2])                          # (B*N, 8, 2)
    ct = c.transpose(1, 2)                          # (B*N, 2, 8)
    ctc = torch.bmm(ct, c)                          # (B*N, 2, 2)
    # NOTE: the build in symeig is slow!
    # _, v = ctc.symeig(eigenvectors=True)
    # v1 = v[:, 0, :].unsqueeze(1)                   
    # v2 = v[:, 1, :].unsqueeze(1)
    v1, v2 = eigenvector_22(ctc)
    v1 = v1.unsqueeze(1)                            # (B*N, 1, 2), eigen value
    v2 = v2.unsqueeze(1)
    p1 = torch.sum(c * v1, dim=-1)                  # (B*N, 8), first principle component
    p2 = torch.sum(c * v2, dim=-1)                  # (B*N, 8), second principle component
    w = p1.max(dim=-1)[0] - p1.min(dim=-1)[0]       # (B*N, ),  width of rotated enclosing box
    h = p2.max(dim=-1)[0] - p2.min(dim=-1)[0]       # (B*N, ),  height of rotated enclosing box
    return w.view([B, -1]), h.view([B, -1])


def eigenvector_22(x:Tensor):
    """return eigenvector of 2x2 symmetric matrix using closed form
    
    https://math.stackexchange.com/questions/8672/eigenvalues-and-eigenvectors-of-2-times-2-matrix
    
    The calculation is done by using double precision

    Args:
        x (Tensor): (..., 2, 2), symmetric, semi-definite
    
    Return:
        v1 (Tensor): (..., 2)
        v2 (Tensor): (..., 2)
    """
    # NOTE: must use doule precision here! with float the back-prop is very unstable
    a = x[..., 0, 0].double()
    c = x[..., 0, 1].double()
    b = x[..., 1, 1].double()                                # (..., )
    delta = torch.sqrt(a*a + 4*c*c - 2*a*b + b*b)
    v1 = (a - b - delta) / 2. /c
    v1 = torch.stack([v1, torch.ones_like(v1, dtype=torch.double, device=v1.device)], dim=-1)    # (..., 2)
    v2 = (a - b + delta) / 2. /c
    v2 = torch.stack([v2, torch.ones_like(v2, dtype=torch.double, device=v2.device)], dim=-1)    # (..., 2)
    n1 = torch.sum(v1*v1, keepdim=True, dim=-1).sqrt()
    n2 = torch.sum(v2*v2, keepdim=True, dim=-1).sqrt()
    v1 = v1 / n1
    v2 = v2 / n2
    return v1.type(x.dtype), v2.type(x.dtype)



    
def generate_table():
    """generate candidates of hull polygon edges and the the other 6 points

    Returns:
        lines: (24, 2)
        points: (24, 6)
    """
    skip = [[0,2], [1,3], [5,7], [4,6]]     # impossible hull edge
    line = []
    points = []

    def all_except_two(o1, o2):
        a = []
        for i in range(8):
            if i != o1 and i != o2:
                a.append(i)
        return a

    for i in range(8):
        for j in range(i+1, 8):
            if [i, j] not in skip:
                line.append([i, j])
                points.append(all_except_two(i, j))
    return line, points


LINES, POINTS = generate_table()
LINES = np.array(LINES).astype(int)
POINTS = np.array(POINTS).astype(int)


def gather_lines_points(corners:Tensor):
    """get hull edge candidates and the rest points using the index

    Args:
        corners (Tensor): (..., 8, 2)
    
    Return: 
        lines (Tensor): (..., 24, 2, 2)
        points (Tensor): (..., 24, 6, 2)
        idx_lines (Tensor): Long (..., 24, 2, 2)
        idx_points (Tensor): Long (..., 24, 6, 2)
    """
    dim = corners.dim()
    idx_lines = torch.LongTensor(LINES).to(corners.device).unsqueeze(-1)      # (24, 2, 1)
    idx_points = torch.LongTensor(POINTS).to(corners.device).unsqueeze(-1)    # (24, 6, 1)
    idx_lines = idx_lines.repeat(1,1,2)                                       # (24, 2, 2)
    idx_points = idx_points.repeat(1,1,2)                                     # (24, 6, 2)
    if dim > 2:
        for _ in range(dim-2):
            idx_lines = torch.unsqueeze(idx_lines, 0)
            idx_points = torch.unsqueeze(idx_points, 0)
        idx_points = idx_points.repeat(*corners.size()[:-2], 1, 1, 1)           # (..., 24, 2, 2)
        idx_lines = idx_lines.repeat(*corners.size()[:-2], 1, 1, 1)             # (..., 24, 6, 2)
    corners_ext = corners.unsqueeze(-3).repeat( *([1]*(dim-2)), 24, 1, 1)       # (..., 24, 8, 2)
    lines = torch.gather(corners_ext, dim=-2, index=idx_lines)                  # (..., 24, 2, 2)
    points = torch.gather(corners_ext, dim=-2, index=idx_points)                # (..., 24, 6, 2)

    return lines, points, idx_lines, idx_points


def point_line_distance_range(lines:Tensor, points:Tensor):
    """calculate the maximal distance between the points in the direction perpendicular to the line
    methode: point-line-distance

    Args:
        lines (Tensor): (..., 24, 2, 2)
        points (Tensor): (..., 24, 6, 2)
    
    Return:
        Tensor: (..., 24)
    """
    x1 = lines[..., 0:1, 0]       # (..., 24, 1)
    y1 = lines[..., 0:1, 1]       # (..., 24, 1)
    x2 = lines[..., 1:2, 0]       # (..., 24, 1)
    y2 = lines[..., 1:2, 1]       # (..., 24, 1)
    x = points[..., 0]            # (..., 24, 6)
    y = points[..., 1]            # (..., 24, 6)
    den = (y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1
    # NOTE: the backward pass of torch.sqrt(x) generates NaN if x==0
    num = torch.sqrt( (y2-y1).square() + (x2-x1).square() + 1e-14 )
    d = den/num         # (..., 24, 6)
    d_max = d.max(dim=-1)[0]       # (..., 24)
    d_min = d.min(dim=-1)[0]       # (..., 24)
    d1 = d_max - d_min             # suppose points on different side
    d2 = torch.max(d.abs(), dim=-1)[0]      # or, all points are on the same side
    # NOTE: if x1 = x2 and y1 = y2, this will return 0
    return torch.max(d1, d2)


def point_line_projection_range(lines:Tensor, points:Tensor):
    """calculate the maximal distance between the points in the direction parallel to the line
    methode: point-line projection

    Args:
        lines (Tensor): (..., 24, 2, 2)
        points (Tensor): (..., 24, 6, 2)
    
    Return:
        Tensor: (..., 24)
    """
    x1 = lines[..., 0:1, 0]       # (..., 24, 1)
    y1 = lines[..., 0:1, 1]       # (..., 24, 1)
    x2 = lines[..., 1:2, 0]       # (..., 24, 1)
    y2 = lines[..., 1:2, 1]       # (..., 24, 1)
    k = (y2 - y1)/(x2 - x1 + 1e-8)      # (..., 24, 1)
    vec = torch.cat([torch.ones_like(k, dtype=k.dtype, device=k.device), k], dim=-1)  # (..., 24, 2)
    vec = vec.unsqueeze(-2)             # (..., 24, 1, 2)
    points_ext = torch.cat([lines, points], dim=-2)         # (..., 24, 8), consider all 8 points
    den = torch.sum(points_ext * vec, dim=-1)               # (..., 24, 8) 
    proj = den / torch.norm(vec, dim=-1, keepdim=False)     # (..., 24, 8)
    proj_max = proj.max(dim=-1)[0]       # (..., 24)
    proj_min = proj.min(dim=-1)[0]       # (..., 24)
    return proj_max - proj_min


def smallest_bounding_box(corners:Tensor, verbose=False):
    """return width and length of the smallest bouding box which encloses two boxes.

    Args:
        lines (Tensor): (..., 24, 2, 2)
        verbose (bool, optional): If True, return area and index. Defaults to False.

    Returns:
        (Tensor): width (..., 24)
        (Tensor): height (..., 24)
        (Tensor): area (..., )
        (Tensor): index of candiatae (..., )
    """
    lines, points, _, _ = gather_lines_points(corners)
    proj = point_line_projection_range(lines, points)   # (..., 24)
    dist = point_line_distance_range(lines, points)     # (..., 24)
    area = proj * dist
    # remove area with 0 when the two points of the line have the same coordinates
    zero_mask = (area == 0).type(corners.dtype)
    fake = torch.ones_like(zero_mask, dtype=corners.dtype, device=corners.device)* 1e8 * zero_mask
    area += fake        # add large value to zero_mask
    area_min, idx = torch.min(area, dim=-1, keepdim=True)     # (..., 1)
    w = torch.gather(proj, dim=-1, index=idx)
    h = torch.gather(dist, dim=-1, index=idx)          # (..., 1)
    w = w.squeeze(-1).type(corners.dtype)
    h = h.squeeze(-1).type(corners.dtype)
    area_min = area_min.squeeze(-1).type(corners.dtype)
    if verbose:
        return w, h, area_min, idx.squeeze(-1)
    else:
        return w, h



if __name__ == "__main__":
    


    import os
    # (339.15, 230.95, 308.218364151133, 33.12346705650146, 1.5331517813945545)
    box1 = torch.tensor([[[339.15, 230.95, 308.218364151133, 33.12346705650146, 1.5331517813945545]]], dtype=torch.float64)
    # box2 = torch.tensor([[[339.15, 230.95, 310.218364151133, 33.12346705650146, 1.5331517813945545]]], dtype=torch.float64)
    box2 = torch.tensor([[[259.15, 230.95, 73.9724073962719, 33.12346705650146, 1.5331517813945545]]], dtype=torch.float64)
    # box2 = torch.tensor([[[339.15, 230.95, 308.218364151133, 33.12346705650146, 1.5331517813945545]]], dtype=torch.float64)

    g_iou = diff_giou_rotated_2d(box1, box2)
    print(f"GIOU: {g_iou.item():.6f}")

    diou = diff_diou_rotated_2d(box1, box2)
    print(f"DIOU: {diou.item():.6f}")

    iou = diff_iou_rotated_2d(box1, box2)
    print(f"IOU: {iou.item():.6f}")
    
    # # 转换为corners用于可视化
    # corners1 = box2corners(box1)
    # corners2 = box2corners(box2)
    
    # # 计算IoU并可视化
    # print("计算IoU...")
    # iou = diff_iou_rotated_2d(box1, box2)
    # print(f"IoU: {iou.item():.6f}")
    
    # # 可视化交集计算过程
    # print("生成可视化...")
    # save_dir = "debug_visualization"
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, "intersection_visualization.png")
    
    # # 调用可视化函数
    # intersection_area, vertices = oriented_box_intersection_2d(
    #     corners1, corners2, 
    #     visualize=True, 
    #     save_path=save_path
    # )
    
    # print(f"交集面积: {intersection_area.item():.6f}")
    # print(f"可视化已保存到: {save_path}")