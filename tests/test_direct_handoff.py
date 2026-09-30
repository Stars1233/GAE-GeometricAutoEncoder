import ast
from pathlib import Path
import types
import numpy as np
from scripts.demo.direct_camera import image_key, end_view_poses
from test_demo_streaming import load_functions


def test_direct_view_is_the_resident_camera_path(tmp_path):
    ns=load_functions()
    tree=ast.parse((Path(__file__).resolve().parents[1]/'camera_studio.py').read_text())
    names={'_camera_preview_path','generate_direct_camera'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns.update(ROOT=tmp_path,OUTPUT_ROOT=tmp_path,HF_REPO="test",CKPT_DIR=tmp_path,
              _pose_reference_for_image=lambda _: None,DEFAULT_EXAMPLE_POSES=tmp_path/"absent")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'camera_studio.py','exec'),ns)
    photo=tmp_path/'source.jpg';photo.write_bytes(b'photo')
    prepared=tmp_path/'prepared.png';prepared.write_bytes(b'prepared pixels')
    scene=dict(imageKey=image_key(photo),pivotDepth=4,K=[[460,0,335],[0,455,189],[0,0,1]],
               preparedImage=str(prepared),metricScale=3)
    payload=dict(schema='camera-path-v2',finalized=True,imageKey=scene['imageKey'],keyframes=[
        dict(role='start',time=0),dict(role='keyframe',time=.3,x=.2,z=.1,yaw=-8),
        dict(role='keyframe',time=.65,x=-.2,y=.1,z=.25,yaw=8),
        dict(role='end',time=1,x=.3,z=.4,yaw=12,pitch=4)])
    received={}
    def generate(image,prompt,**kwargs):
        received.update(kwargs)
        received["image"]=image
        return dict(timings=dict(request_seconds=1),rgb_video='rgb',synchronized_video='paired',geometry='geometry')
    ns['RESIDENT_ENGINE']=types.SimpleNamespace(generate=generate)
    list(ns['generate_direct_camera'](str(photo),'scene','direct-view',81,25,2,42,2,payload,scene))
    saved=np.load(received['poses'])
    np.testing.assert_array_equal(saved['c2w'],end_view_poses(payload,scene,81))
    np.testing.assert_allclose(saved['K'][0],scene['K'],atol=1e-4)
    assert received['image']==str(prepared) and bool(saved['image_preprocessed'])
    assert float(saved['metric_scale'])==3
    assert received['views']==81 and received['steps']==25
