"""Direct end-view camera poses in the model's +Z forward, +Y down gauge."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def image_key(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def end_view_poses(payload, scene, views):
    params = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(params, dict) or params.get('imageKey') != scene.get('imageKey'):
        raise ValueError('Wait for this image to finish preparing its camera preview.')
    n = int(views)
    if n < 2 or n > 161:
        raise ValueError('Camera video requires 2–161 views.')
    depth = float(scene['pivotDepth'])
    cartesian = params.get('schema') == 'camera-path-v2'
    if cartesian and params.get('finalized') is not True:
        raise ValueError('Choose the final camera before generating.')
    names = ('yaw', 'pitch', 'x', 'y', 'z') if cartesian else ('yaw', 'pitch', 'dolly', 'slideX', 'slideY')
    defaults = (0, 0, 0, 0, 0) if cartesian else (0, 0, 1, 0, 0)
    keys = params.get('keyframes')
    if keys is None:
        keys = [dict(zip(names, defaults)), params]
    if not isinstance(keys, list) or not 2 <= len(keys) <= 6 or not all(isinstance(k, dict) for k in keys):
        raise ValueError('Use the original view, an ending view, and up to four intermediate keyframes.')
    if cartesian and (keys[0].get('role') != 'start' or keys[-1].get('role') != 'end'
                      or any(k.get('role') != 'keyframe' for k in keys[1:-1])):
        raise ValueError('Use a fixed start, saved keyframes, and an explicit final camera.')
    values = np.array([[k.get(name, default) for name, default in zip(names, defaults)] for k in keys], float)
    if not np.all(np.isfinite(values)) or not np.isfinite(depth) or depth <= 0:
        raise ValueError('Invalid camera view.')
    yaw_limit, pitch_limit = (3600.001, 85.001) if cartesian else (35.001, 20.001)
    if np.any(np.abs(values[:,0]) > yaw_limit) or np.any(np.abs(values[:,1]) > pitch_limit):
        raise ValueError('Camera view exceeds the supported rotation range.')
    if cartesian:
        if np.any(np.abs(values[:,2:]) > 10.0001 * depth):
            raise ValueError('Camera position exceeds the supported movement range.')
    elif (np.any(values[:,2] < .4999) or np.any(values[:,2] > 1.6001)
          or np.any(np.abs(values[:,3:]) > .5001 * depth)):
        raise ValueError('Camera view exceeds the supported movement range.')
    if not np.allclose(values[0], defaults, atol=1e-8, rtol=0):
        raise ValueError('The first keyframe must be the original photo view.')
    times = np.array([k.get('time', i/(len(keys)-1)) for i,k in enumerate(keys)], float)
    if (not np.all(np.isfinite(times)) or times[0] != 0 or times[-1] != 1
        or np.any(np.diff(times) <= 0)):
        raise ValueError('Keyframe times must increase from the original view to the end.')
    # Every requested key camera appears as an actual generated frame.
    indices = np.floor(times * (n-1) + .5).astype(int)
    if np.any(np.diff(indices) <= 0):
        raise ValueError('These keyframes are too close. Spread them out or use more video frames.')
    sampled = np.empty((n,5), float)
    tangents = np.zeros_like(values)
    continuous = cartesian and params.get('interpolation') == 'pchip'
    if continuous:
        intervals = np.diff(indices).astype(float)
        slopes = np.diff(values, axis=0) / intervals[:, None]
        for j in range(1, len(keys)-1):
            left, right = slopes[j-1], slopes[j]
            same = left * right > 0
            w1, w2 = 2*intervals[j]+intervals[j-1], intervals[j]+2*intervals[j-1]
            tangents[j, same] = (w1+w2)/(w1/left[same]+w2/right[same])
    for j, (a,b) in enumerate(zip(indices[:-1],indices[1:])):
        u = np.linspace(0,1,b-a+1)
        if continuous:
            h = b-a
            sampled[a:b+1] = ((2*u**3-3*u**2+1)[:,None]*values[j]
                + (u**3-2*u**2+u)[:,None]*h*tangents[j]
                + (-2*u**3+3*u**2)[:,None]*values[j+1]
                + (u**3-u**2)[:,None]*h*tangents[j+1])
        else:
            u = u*u*(3-2*u)
            sampled[a:b+1] = values[j] + u[:,None]*(values[j+1]-values[j])
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    for i, (yaw,pitch,a,b,c) in enumerate(sampled):
        y, p = np.deg2rad([yaw, pitch])
        ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
        rx = np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
        r = ry @ rx
        poses[i,:3,:3] = r
        poses[i,:3,3] = [a, b, c] if cartesian else [b, c, depth] - r @ np.array([0,0,depth*a])
    return poses.astype(np.float32)


def metric_depth_scale(relative_z, metric_prediction, K, sky=None):
    """One robust scene scale; metric head follows DA3's focal / 300 rule."""
    relative = np.asarray(relative_z, float)
    metric = np.asarray(metric_prediction, float) * float((K[0, 0] + K[1, 1]) / 600)
    valid = np.isfinite(relative) & np.isfinite(metric) & (relative > .01) & (metric > .01)
    if sky is not None:
        valid &= np.asarray(sky).reshape(relative.shape) < .3
    if np.count_nonzero(valid) < 100:
        raise ValueError("Too little valid geometry for metric camera calibration.")
    ratios = metric[valid] / relative[valid]
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid metric camera calibration.")
    return scale, np.percentile(ratios, [10, 50, 90]).tolist()
