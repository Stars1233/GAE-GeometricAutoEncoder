import numpy as np
import pytest
from scripts.demo.direct_camera import end_view_poses

SCENE = dict(imageKey='scene', pivotDepth=4)


def test_identity_and_orbit_keep_lookat_center():
    static = end_view_poses(dict(imageKey='scene'), SCENE, 81)
    np.testing.assert_array_equal(static, np.repeat(np.eye(4)[None],81,axis=0))
    poses = end_view_poses(dict(imageKey='scene',yaw=30,pitch=12,dolly=.8),SCENE,81)
    np.testing.assert_array_equal(poses[0],np.eye(4))
    r = poses[:,:3,:3]
    np.testing.assert_allclose(r @ r.transpose(0,2,1),np.broadcast_to(np.eye(3),r.shape),atol=2e-7)
    target = (np.array([0,0,4])-poses[-1,:3,3]) @ r[-1]
    np.testing.assert_allclose(target,[0,0,3.2],atol=2e-6)
    assert np.max(np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)) < .06


def test_dolly_is_camera_translation_with_fixed_orientation():
    poses=end_view_poses(dict(imageKey='scene',dolly=.5,slideX=.3),SCENE,81)
    np.testing.assert_array_equal(poses[:,:3,:3],np.repeat(np.eye(3)[None],81,axis=0))
    np.testing.assert_allclose(poses[-1,:3,3],[.3,0,2])


@pytest.mark.parametrize('params',[dict(imageKey='other'),dict(imageKey='scene',yaw=36),dict(imageKey='scene',dolly=float('nan')),dict(imageKey='scene',slideY=2.1)])
def test_invalid_or_stale_view_is_rejected(params):
    with pytest.raises(ValueError):end_view_poses(params,SCENE,81)


def test_three_segments_hit_intermediate_cameras_and_keep_rotations_valid():
    keys=[dict(time=0),dict(time=1/3,yaw=-15,dolly=.9,slideY=.2),
          dict(time=2/3,yaw=20,pitch=8,slideX=.3),dict(time=1,yaw=3,dolly=1.1)]
    poses=end_view_poses(dict(imageKey='scene',keyframes=keys),SCENE,81)
    for index,key in zip([0,27,53,80],keys):
        expected=end_view_poses(dict(imageKey='scene',**{k:v for k,v in key.items() if k!='time'}),SCENE,2)[-1]
        np.testing.assert_allclose(poses[index],expected,atol=1e-6)
    rotations=poses[:,:3,:3]
    np.testing.assert_allclose(np.linalg.det(rotations),np.ones(81),atol=2e-7)
    np.testing.assert_allclose(rotations @ rotations.transpose(0,2,1),np.broadcast_to(np.eye(3),rotations.shape),atol=2e-7)


@pytest.mark.parametrize('views',[17,33,81])
def test_explicit_key_times_remap_to_real_frames(views):
    keys=[dict(time=0),dict(time=.25,yaw=-12),dict(time=.75,yaw=18),dict(time=1,dolly=.85)]
    poses=end_view_poses(dict(imageKey='scene',keyframes=keys),SCENE,views)
    for key in keys:
        expected=end_view_poses(dict(imageKey='scene',**{k:v for k,v in key.items() if k!='time'}),SCENE,2)[-1]
        np.testing.assert_allclose(poses[int(key['time']*(views-1)+.5)],expected,atol=1e-6)


@pytest.mark.parametrize('keys',[ [dict(time=0,yaw=1),dict(time=1)],
    [dict(time=0),dict(time=.5),dict(time=.5),dict(time=1)],
    [dict(time=0),dict(time=.001),dict(time=1)],
    [dict(time=0),dict(time=float('nan')),dict(time=1)],
    [dict(time=0),dict(time=1.1),dict(time=1)] ])
def test_invalid_keyframe_timeline_is_rejected(keys):
    with pytest.raises(ValueError):end_view_poses(dict(imageKey='scene',keyframes=keys),SCENE,81)


def test_metric_scale_uses_focal_and_excludes_sky():
    from scripts.demo.direct_camera import metric_depth_scale
    relative = np.full((20,20), 2.)
    prediction = np.full((20,20), 3.)
    sky = np.zeros((20,20)); sky[:10] = 1
    prediction[:10] = 30000
    K = np.diag([600.,600.,1.])
    scale, quantiles = metric_depth_scale(relative, prediction, K, sky)
    assert scale == 3 and quantiles == [3,3,3]
    # Scaling all geometry and translations preserves the preview projection.
    a = end_view_poses(dict(imageKey='x',yaw=10),dict(imageKey='x',pivotDepth=2),81)
    b = end_view_poses(dict(imageKey='x',yaw=10),dict(imageKey='x',pivotDepth=6),81)
    np.testing.assert_allclose(b[:,:3,3],a[:,:3,3]*3,atol=1e-6)
    np.testing.assert_array_equal(b[:,:3,:3],a[:,:3,:3])


def test_camera_studio_requires_explicit_final_and_keeps_aim_position():
    scene=dict(imageKey='scene',pivotDepth=4)
    keys=[dict(role='start',time=0),dict(role='end',time=1,x=.3,y=.2,z=.6,yaw=20,pitch=5)]
    payload=dict(schema='camera-path-v2',imageKey='scene',keyframes=keys)
    with pytest.raises(ValueError,match='final camera'):
        end_view_poses(payload,scene,81)
    payload['finalized']=True
    poses=end_view_poses(payload,scene,81)
    np.testing.assert_allclose(poses[-1,:3,3],[.3,.2,.6])
    keys[-1]['yaw']=-20
    np.testing.assert_array_equal(end_view_poses(payload,scene,81)[:,:3,3],poses[:,:3,3])
    keys[-1]['role']='keyframe'
    with pytest.raises(ValueError,match='explicit final'):
        end_view_poses(payload,scene,81)


def test_continuous_camera_path_has_no_artificial_stop_or_overshoot():
    payload=dict(schema='camera-path-v2',interpolation='pchip',finalized=True,imageKey='scene',
        keyframes=[dict(role='start',time=0),dict(role='keyframe',time=.5,x=.4,yaw=8),dict(role='end',time=1,x=.8,yaw=16)])
    path=end_view_poses(payload,dict(imageKey='scene',pivotDepth=4),81)
    assert np.all(np.diff(path[:,0,3])>=0)
    assert path[:,0,3].min()==0 and path[:,0,3].max()<=.800001
    assert path[40,0,3]-path[39,0,3]>.005
    assert abs((path[40,0,3]-path[39,0,3])-(path[41,0,3]-path[40,0,3]))<1e-5
    np.testing.assert_allclose(path[[0,40,80],0,3],[0,.4,.8],atol=1e-6)


def test_studio_accepts_long_forward_path_and_full_rotation():
    payload=dict(schema='camera-path-v2',interpolation='pchip',finalized=True,imageKey='scene',
        keyframes=[dict(role='start',time=0),dict(role='keyframe',time=.5,z=8,yaw=180,pitch=60),dict(role='end',time=1,z=20,yaw=360,pitch=0)])
    poses=end_view_poses(payload,dict(imageKey='scene',pivotDepth=4),81)
    np.testing.assert_allclose(poses[-1,:3,3],[0,0,20])
    assert np.all(np.isfinite(poses))
    np.testing.assert_allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-6)
