# A small window whose content changes every 4 ms, so screen-capture benchmarks see new frames.
import tkinter as tk, time, sys
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
root = tk.Tk(); root.title("ganglion-animate"); root.geometry("400x300+100+100")
c = tk.Canvas(root, width=400, height=300, bg="black"); c.pack()
rect = c.create_rectangle(0, 0, 400, 300, fill="#000000", outline="")
t0 = time.perf_counter(); i = 0
def tick():
    global i
    i += 1
    c.itemconfig(rect, fill="#%02x%02x%02x" % ((i*7) % 256, (i*13) % 256, (i*29) % 256))
    if time.perf_counter() - t0 < dur: root.after(4, tick)
    else: root.destroy()
root.after(4, tick); root.mainloop()
