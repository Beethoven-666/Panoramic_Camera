"""Independent, read-only TSDF delivery for a published video panorama."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .config import load_config
from .dense_fusion import export_tsdf_mesh_pair
from .rgbd_projection import PinholeIntrinsics
from .video_session import load_video_session


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_2d_report(output: Path) -> dict[str, Any]:
    marker = output / "video_delivery.json"
    report_path = output / "video_report.json"
    if not marker.is_file() or not report_path.is_file():
        raise ValueError("g305-video-3d requires an already published video 2-D delivery")
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    if marker_value.get("schema") != "gemini305-video-panorama-delivery/v1":
        raise ValueError("Video 2-D delivery marker has an unsupported schema")
    if marker_value.get("delivery_state") not in {"published", "published_degraded"}:
        raise ValueError("Video 2-D delivery is not published")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _invalidate_3d(output: Path) -> None:
    for name in ("video_3d_delivery.json", "video_3d_failure.json", "video_tsdf_mesh.glb", "video_tsdf_mesh_mobile.glb", "video_tsdf_mesh_viewer.html"):
        (output / name).unlink(missing_ok=True)


def _offline_glb_viewer(mesh_filename: str = "video_tsdf_mesh.glb") -> str:
    """Return a dependency-free WebGL 1 viewer for the sibling GLB.

    ``model-viewer`` is a custom element and the former page neither loaded it
    nor bundled it, leaving a blank page.  This small viewer intentionally uses
    only browser APIs, so opening the delivery on an offline inspection PC is
    supported as well.
    """
    return f'''<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemini 305 video TSDF mesh</title>
<style>html,body,canvas{{margin:0;width:100%;height:100%;overflow:hidden;background:#16181d}}#note{{position:fixed;left:12px;top:10px;color:#eef2ff;font:14px system-ui;background:#0009;padding:7px 9px;border-radius:4px}}</style>
<canvas id="mesh"></canvas><div id="note">Loading mesh…</div>
<script>
(() => {{
 const canvas=document.querySelector('#mesh'), note=document.querySelector('#note'), gl=canvas.getContext('webgl',{{antialias:true}});
 if(!gl){{note.textContent='WebGL is unavailable in this browser.';return;}}
 if(!gl.getExtension('OES_element_index_uint')){{note.textContent='This browser lacks 32-bit WebGL mesh-index support.';return;}}
 const vs=`attribute vec3 p; attribute vec4 c; uniform mat4 m; varying vec4 v; void main(){{gl_Position=m*vec4(p,1.);v=c;}}`, fs=`precision mediump float; varying vec4 v; void main(){{gl_FragColor=v;}}`;
 const shader=(t,s)=>{{let x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);if(!gl.getShaderParameter(x,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(x));return x;}};
 const pg=gl.createProgram();gl.attachShader(pg,shader(gl.VERTEX_SHADER,vs));gl.attachShader(pg,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(pg);if(!gl.getProgramParameter(pg,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(pg));gl.useProgram(pg);
 const loc={{p:gl.getAttribLocation(pg,'p'),c:gl.getAttribLocation(pg,'c'),m:gl.getUniformLocation(pg,'m')}};
 const type={{5120:Int8Array,5121:Uint8Array,5122:Int16Array,5123:Uint16Array,5125:Uint32Array,5126:Float32Array}};
 const comp={{5120:1,5121:1,5122:2,5123:2,5125:4,5126:4}}, count={{SCALAR:1,VEC2:2,VEC3:3,VEC4:4}};
 const mat=(a,b)=>{{let r=new Float32Array(16);for(let c=0;c<4;c++)for(let i=0;i<4;i++)r[c*4+i]=a[i]*b[c*4]+a[4+i]*b[c*4+1]+a[8+i]*b[c*4+2]+a[12+i]*b[c*4+3];return r;}};
 const proj=(f,a,n,z)=>{{let q=1/Math.tan(f/2),r=new Float32Array(16);r[0]=q/a;r[5]=q;r[10]=(z+n)/(n-z);r[11]=-1;r[14]=2*z*n/(n-z);return r;}};
 const view=(yaw,pitch,d)=>{{let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);return new Float32Array([cy,sp*sy,-cp*sy,0,0,cp,sp,0,sy,-sp*cy,cp*cy,0,0,0,-d,1]);}};
 let buffers, index, elements, yaw=.55, pitch=-.25, distance=3;
 function draw(){{if(!buffers)return;let w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*devicePixelRatio;canvas.height=h*devicePixelRatio;gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(.086,.094,.114,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.enableVertexAttribArray(loc.p);gl.bindBuffer(gl.ARRAY_BUFFER,buffers.p);gl.vertexAttribPointer(loc.p,3,gl.FLOAT,false,0,0);if(buffers.c){{gl.enableVertexAttribArray(loc.c);gl.bindBuffer(gl.ARRAY_BUFFER,buffers.c);gl.vertexAttribPointer(loc.c,4,gl.UNSIGNED_BYTE,true,0,0);}}else{{gl.disableVertexAttribArray(loc.c);gl.vertexAttrib4f(loc.c,.75,.82,.9,1);}}gl.uniformMatrix4fv(loc.m,false,mat(proj(1.0,w/h,.01,100),view(yaw,pitch,distance)));gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,index);gl.drawElements(gl.TRIANGLES,elements,gl.UNSIGNED_INT,0);}}
 fetch('{mesh_filename}').then(r=>{{if(!r.ok)throw Error('Could not load '+r.status);return r.arrayBuffer();}}).then(raw=>{{let dv=new DataView(raw), jl=dv.getUint32(12,true), json=JSON.parse(new TextDecoder().decode(new Uint8Array(raw,20,jl))), off=20+jl+8, bin=raw.slice(off), prim=json.meshes[0].primitives[0], get=i=>{{let a=json.accessors[i],v=json.bufferViews[a.bufferView],T=type[a.componentType],start=(v.byteOffset||0)+(a.byteOffset||0);return {{a,v,T,data:new T(bin,start,a.count*count[a.type])}}}};let pos=get(prim.attributes.POSITION), col=prim.attributes.COLOR_0===undefined?null:get(prim.attributes.COLOR_0), ind=get(prim.indices), min=pos.a.min,max=pos.a.max,span=Math.max(.001,Math.hypot(max[0]-min[0],max[1]-min[1],max[2]-min[2])),cx=(min[0]+max[0])/2,cy=(min[1]+max[1])/2,cz=(min[2]+max[2])/2;for(let i=0;i<pos.a.count;i++){{pos.data[3*i]=(pos.data[3*i]-cx)/span;pos.data[3*i+1]=-(pos.data[3*i+1]-cy)/span;pos.data[3*i+2]=-(pos.data[3*i+2]-cz)/span;}}buffers={{p:gl.createBuffer(),c:col&&gl.createBuffer()}};gl.bindBuffer(gl.ARRAY_BUFFER,buffers.p);gl.bufferData(gl.ARRAY_BUFFER,pos.data,gl.STATIC_DRAW);if(col){{let rgba=new Uint8Array(pos.a.count*4);for(let i=0;i<pos.a.count;i++)for(let j=0;j<3;j++)rgba[4*i+j]=col.data[i*count[col.a.type]+j]*(col.a.componentType===5126?255:1);for(let i=0;i<pos.a.count;i++)rgba[4*i+3]=255;gl.bindBuffer(gl.ARRAY_BUFFER,buffers.c);gl.bufferData(gl.ARRAY_BUFFER,rgba,gl.STATIC_DRAW);}}index=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,index);let ix=ind.data instanceof Uint32Array?ind.data:new Uint32Array(ind.data);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,ix,gl.STATIC_DRAW);elements=ix.length;distance=1.6;note.textContent='Drag to rotate · wheel to zoom';draw();}}).catch(e=>note.textContent='Mesh error: '+e.message);
 let last;canvas.onpointerdown=e=>{{last=e;canvas.setPointerCapture(e.pointerId)}};canvas.onpointermove=e=>{{if(!last)return;yaw+=(e.clientX-last.clientX)*.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-last.clientY)*.01));last=e;draw()}};canvas.onpointerup=()=>last=null;canvas.onwheel=e=>{{distance*=Math.exp(e.deltaY*.001);draw();e.preventDefault()}};addEventListener('resize',draw);
}})();
</script>'''


def publish_video_3d(output: str | Path, *, input_path: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Produce video GLBs without changing any 2-D delivery file."""
    root = Path(output).expanduser().resolve()
    source = Path(input_path).expanduser().resolve()
    _invalidate_3d(root)
    try:
        report = _load_2d_report(root)
        session = load_video_session(source)
        expected = report.get("input_sha256")
        actual = {"manifest": _sha256(session.rgbd.root / "manifest.json"), "calibration": _sha256(session.rgbd.root / "calibration.json")}
        if expected != actual:
            raise ValueError("Video source manifest/calibration no longer match its published 2-D delivery")
        ids = report.get("source_frame_ids")
        orb = report.get("orbslam3", {})
        if not isinstance(ids, list) or not isinstance(orb, dict):
            raise ValueError("Video 2-D report does not contain audited real sources")
        pose_map = {int(frame_id): pose for frame_id, pose in zip(orb.get("tracked_frame_ids", []), orb.get("camera_to_world", []), strict=True)}
        frames = tuple(frame for frame in session.rgbd.frames if frame.frame_id in set(ids))
        if [frame.frame_id for frame in frames] != ids or any(frame_id not in pose_map for frame_id in ids):
            raise ValueError("Video 3-D sources lack complete genuine ORB poses")
        intrinsics = session.rgbd.calibration
        camera = PinholeIntrinsics(width=intrinsics.width, height=intrinsics.height, fx=intrinsics.fx, fy=intrinsics.fy, cx=intrinsics.cx, cy=intrinsics.cy, distortion=intrinsics.distortion)
        tsdf = dict((config or load_config(None)).get("stitch", {}).get("tsdf_visualization", {}))
        desktop, mobile, audit = export_tsdf_mesh_pair(frames, [pose_map[frame_id] for frame_id in ids], camera, config=tsdf)
        (root / ".video_tsdf_mesh.pending.glb").write_bytes(desktop)
        (root / ".video_tsdf_mesh_mobile.pending.glb").write_bytes(mobile)
        (root / ".video_tsdf_mesh_viewer.pending.html").write_text(_offline_glb_viewer(), encoding="utf-8")
        marker = {"schema": "gemini305-video-3d-delivery/v1", "delivery_state": "published", "mesh": "video_tsdf_mesh.glb", "mobile_mesh": "video_tsdf_mesh_mobile.glb", "viewer": "video_tsdf_mesh_viewer.html", "audit": audit}
        (root / ".video_3d_delivery.pending.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
        os.replace(root / ".video_tsdf_mesh.pending.glb", root / "video_tsdf_mesh.glb")
        os.replace(root / ".video_tsdf_mesh_mobile.pending.glb", root / "video_tsdf_mesh_mobile.glb")
        os.replace(root / ".video_tsdf_mesh_viewer.pending.html", root / "video_tsdf_mesh_viewer.html")
        os.replace(root / ".video_3d_delivery.pending.json", root / "video_3d_delivery.json")
        return marker
    except Exception as exc:
        pending = root / ".video_3d_failure.pending.json"
        pending.write_text(json.dumps({"schema": "gemini305-video-3d-failure/v1", "error_type": type(exc).__name__, "message": str(exc), "two_d_delivery_preserved": (root / "video_delivery.json").is_file()}, indent=2), encoding="utf-8")
        os.replace(pending, root / "video_3d_failure.json")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GLB artifacts for an existing video 2-D delivery")
    parser.add_argument("output", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        publish_video_3d(args.output, input_path=args.input, config=config)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
