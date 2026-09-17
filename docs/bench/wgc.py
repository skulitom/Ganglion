import time, numpy as np
from windows_capture import WindowsCapture, Frame, InternalCaptureControl
cnt={"n":0,"t0":None,"stamps":[]}
capture = WindowsCapture(cursor_capture=False, draw_border=False, monitor_index=1)
@capture.event
def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
    now=time.perf_counter()
    if cnt["t0"] is None: cnt["t0"]=now
    cnt["n"]+=1; cnt["stamps"].append(now)
    if cnt["n"]==1: print("frame:", frame.width, frame.height, type(frame.frame_buffer))
    if now-cnt["t0"]>2.0: capture_control.stop()
@capture.event
def on_closed(): pass
capture.start()
st=np.diff(cnt["stamps"])*1000
print(f"windows-capture (WGC): {cnt['n']} frames in {cnt['stamps'][-1]-cnt['t0']:.2f} s -> {cnt['n']/(cnt['stamps'][-1]-cnt['t0']):.0f} fps; inter-frame median {np.median(st):.2f} ms, p95 {np.percentile(st,95):.2f} ms")
