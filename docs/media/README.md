# How the clips are made

Every clip has the fly brain on the left and the application it is steering on the right. The
two halves come from one run and are joined afterwards, on the ledger's clock.

**The application.** `ganglion.viz.screenrec` runs beside the core, in the session the
application lives in, and saves the screen as JPEG frames stamped with `time.monotonic()`, the
clock the ledger uses. The headless reach demo has no window to film; `ganglion.viz.reachclip`
draws its scene again from the demo's own record (the evaluator's truth, the pointer steps, and
the target's path, which is a function of the trial's seed and time).

**The brain.** Nothing is recorded from the network while it runs. The ledger holds every
sample the model was given (`shadow_prediction` and `shadow_discarded` events carry the sample,
the reset flag and the number of neural steps), so `ganglion.viz.brainvideo` runs the same
checkpoint over the same samples again and reads the firing rates out. It checks itself: the
proposals of the second run are compared with the logged ones, and the largest difference is
printed (0.0007 for the reach clip, in units of the intent speed). Each neuron is drawn at its
soma position in the male CNS (Haltere's layout: brain on top, nerve cord below), coloured by
population, and brightened by how far its rate is above its own mean over the clip. While no
motor program runs the model does not run either, and the panel shows the brain at rest.

```powershell
# the headless reach clip: needs Haltere, torch with CUDA, Pillow and ffmpeg, but no desktop
.venv/Scripts/python.exe -m ganglion.cli reach-demo --environment synthetic --trials 6 --controller connectome --shadow-checkpoint <checkpoint> --shadow-process --no-home --json runs/video/reach.json
.venv/Scripts/python.exe -m ganglion.viz.reachclip runs/video/reach.json --out runs/video/reach
.venv/Scripts/python.exe -m ganglion.viz.brainvideo --events runs/video/reach/events.jsonl --frames runs/video/reach --checkpoint <checkpoint> --out docs/media/reach --title "fly brain moving a cursor to a moving target" --start 0.1 --seconds 13 --height 360

# a live application: record beside the core, then compose from the pilot's or the client's ledger
.venv/Scripts/python.exe -m ganglion.viz.screenrec --out runs/video/fight --seconds 25 --fps 20 --region 0,0,1280,720
.venv/Scripts/python.exe -m ganglion.viz.brainvideo --events runs/hl/pilot/events.jsonl --frames runs/video/fight --checkpoint <checkpoint> --out docs/media/halflife --title "fly brain steering the view in Half-Life"
```

The checkpoint is `cursor-readout-v6.pt` from the
[model repository](https://huggingface.co/Skulitom/ganglion-haltere-cursor).
