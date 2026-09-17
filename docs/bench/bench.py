import time, sys
import numpy as np
try:
    import dxcam
    cam = dxcam.create(output_idx=0, output_color="BGR")
    print("dxcam outputs:", dxcam.device_info(), dxcam.output_info())
    f = cam.grab();
    t0=time.perf_counter(); n=0; none=0; t_end=t0+2.0
    while time.perf_counter()<t_end:
        f=cam.grab()
        if f is None: none+=1
        else: n+=1
    print(f"dxcam grab(): {n} new frames in 2 s ({n/2:.0f} fps of changed frames), {none} no-change polls; frame shape {None if f is None else f.shape}")
    # timed single grabs after a change (move nothing; measure call cost)
    ts=[]
    for _ in range(50):
        t=time.perf_counter(); f=cam.grab(); ts.append((time.perf_counter()-t)*1000)
    print(f"dxcam grab() call cost: median {np.median(ts):.2f} ms, max {max(ts):.2f} ms")
    cam.start(target_fps=240, video_mode=True)
    time.sleep(0.5); t=time.perf_counter(); n=0
    while time.perf_counter()-t<2.0:
        f=cam.get_latest_frame(); n+=1
    cam.stop()
    print(f"dxcam video_mode 240 target: get_latest_frame loop {n/2:.0f}/s (blocks until a frame; video_mode repeats frames)")
    del cam
except Exception as e:
    print("dxcam failed:", type(e).__name__, e)

try:
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    cnt={"n":0,"t0":None,"t_last":None,"stamps":[]}
    capture = WindowsCapture(cursor_capture=False, draw_border=False, monitor_index=1)
    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        now=time.perf_counter()
        if cnt["t0"] is None: cnt["t0"]=now
        cnt["n"]+=1; cnt["stamps"].append(now)
        if now-cnt["t0"]>2.0:
            capture_control.stop()
    @capture.event
    def on_closed():
        pass
    capture.start()
    st=np.diff(cnt["stamps"])*1000 if len(cnt["stamps"])>2 else np.array([0])
    print(f"windows-capture (WGC): {cnt['n']} frames in {cnt['stamps'][-1]-cnt['t0']:.2f} s -> {cnt['n']/max(1e-6,(cnt['stamps'][-1]-cnt['t0'])):.0f} fps; inter-frame median {np.median(st):.2f} ms")
except Exception as e:
    print("windows-capture failed:", type(e).__name__, e)
