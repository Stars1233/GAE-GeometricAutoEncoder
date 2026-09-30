"""Browser timer must not mistake a previous/cached video for a new result."""

import ast
import shutil
import subprocess
from pathlib import Path

import pytest


def test_timer_waits_for_new_result_even_with_identical_video_url(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable")
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "camera_studio.py").read_text())
    js = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PLAYBACK_TIMING_JS" for t in n.targets)
    )
    harness = r"""
const assert = require('node:assert/strict');
const busy=[];const window={gaeCameraEditor:{setGenerating:value=>busy.push(value)}};
let now = 0, inspect, oldLoaded, tick, intervalCleared=false;
const metric = {dataset:{},textContent:''};
const bar = {hidden:true,value:0};
const makeVideo = () => ({currentSrc:'same-content.mp4',readyState:4,
  getAttribute:()=> 'same-content.mp4',addEventListener:(_event,fn)=>{if(!oldLoaded)oldLoaded=fn;}});
let video = makeVideo();
const container = {querySelector:()=>video};
const document = {getElementById:id=>id==='gae-client-latency'?metric:(id==='gae-request-progress'?bar:container)};
const performance = {now:()=>now};
class MutationObserver { constructor(fn){inspect=fn;} observe(){} disconnect(){} }
const requestAnimationFrame = fn=>fn();
const setTimeout = ()=>1, clearTimeout = ()=>{};
const setInterval = fn=>{tick=fn;return 2;}, clearInterval = ()=>{intervalCleared=true;};
const measure = TIMER;
assert.deepEqual(measure(81,25),[81,25]);
assert.deepEqual(busy,[true]);
assert.match(metric.textContent,/Estimated ready in about 40 s/);
assert.equal(bar.hidden,false);
now=1000; inspect();
assert.match(metric.textContent,/Estimated ready/);
video=null; inspect();
oldLoaded();
assert.match(metric.textContent,/Estimated ready/);
now=10000; tick();
assert.match(metric.textContent,/about 30 s/);
assert.equal(bar.value,25);
now=41000; tick();
assert.match(metric.textContent,/initial estimate exceeded/);
assert.equal(bar.value,95);
now=2345; video=makeVideo(); inspect();
assert.match(metric.textContent,/Ready to play in 2.35 s/);
assert.equal(bar.value,100);
assert.equal(intervalCleared,true);
assert.deepEqual(busy,[true,false]);
""".replace("TIMER", js)
    path = tmp_path / "timer.cjs"
    path.write_text(harness)
    subprocess.run([node, str(path)], check=True, capture_output=True, text=True)
