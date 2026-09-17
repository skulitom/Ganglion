import time, sys, ctypes
import numpy as np
print("session/screen:", ctypes.windll.user32.GetSystemMetrics(0), "x", ctypes.windll.user32.GetSystemMetrics(1), flush=True)
try:
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[1]; print("mss monitor:", mon, flush=True)
        region = {"left":mon["left"],"top":mon["top"],"width":640,"height":360}
        sct.grab(region); t=time.perf_counter(); n=30
        for _ in range(n): sct.grab(region)
        print(f"mss GDI grab 640x360: {(time.perf_counter()-t)/n*1000:.2f} ms/frame", flush=True)
except Exception as e:
    print("mss failed:", type(e).__name__, e, flush=True)
try:
    import dxcam
    print("dxcam devices:", dxcam.device_info().strip(), "| outputs:", dxcam.output_info().strip(), flush=True)
    cam = dxcam.create(output_idx=0, output_color="BGR")
    f = cam.grab(); print("first grab:", None if f is None else f.shape, flush=True)
    t0=time.perf_counter(); n=0; none=0
    while time.perf_counter()-t0<3.0:
        f=cam.grab()
        if f is None: none+=1
        else: n+=1
    print(f"dxcam grab(): {n} new frames in 3 s ({n/3:.0f} fps of changed frames), {none} no-change polls", flush=True)
    ts=[]
    for _ in range(50):
        t=time.perf_counter(); f=cam.grab(); ts.append((time.perf_counter()-t)*1000)
    print(f"dxcam grab() call cost: median {np.median(ts):.2f} ms, max {max(ts):.2f} ms", flush=True)
    del cam
except Exception as e:
    print("dxcam failed:", type(e).__name__, e, flush=True)
