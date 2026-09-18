# Half-Life through the reflex layer

First live first-person test, 2026-09-17, inside the seat at 1280×720 (Half-Life, Steam
app 70, build 15961492, OpenGL, `m_rawinput 1`). The core bound the game window by process id
and ran with the DAgger cursor checkpoint loaded. An agent-driven pilot
(`ganglion.evaluation.halflife.pilot`) kept one lease alive and issued high-level commands;
everything at frame rate happened in the core.

## What the core does now

- **Keys, relative look and button holds** as bounded inputs: `hold` renews up to four keys
  (never past one second per command), `look` sends relative mouse counts, optionally spread
  over chunks 10 ms apart, `button` holds a mouse button without moving the pointer. The
  independent helper releases all of them on deadline, halt or producer death.
- **Motion watches** report the largest blob that changed against a frame 60 ms older, on a
  ratio-to-local-brightness image so flashing lights cancel, only after a short persistence, and
  never while the runtime's own commands move the view or the player.
- **Track watches** follow a crop by normalised cross-correlation in a search window, adapting
  the template slowly and reporting loss after repeated misses. A `track` reflex cuts the
  template around the blob a motion watch just found, so the turn itself no longer blinds the
  program.
- **Align** turns the view with relative deltas until the tracked target sits at the crosshair,
  then holds fire in bursts; with `controller: connectome` the model proposes the view velocity
  every tick against a virtual cursor/goal pair and the same supervising envelope as reach.
- **Move** holds keys continuously, renewed every 100 ms by the runtime, until a condition,
  duration, cancellation or fault; it runs beside a pointer program.
- A whole-view **change** energy in every snapshot lets a pilot notice it is blocked.

## Recorded engagement

Two fights against grunts spawned with the console (`give monster_human_grunt`: one, then
three at once) in the corridor before Silo D, with the motion → track → connectome align → fire
chain armed and the agent not aiming. The [ledger summary](results/halflife/grunt-fights-2026-09-17.json)
covers the whole pilot session, including the false alarms on the elevator door and the alarm
light before the percept was made illumination-invariant.

| Measurement | Pilot session |
|---|---:|
| Reflex firings / align intents | 58 / 58 |
| Intents that aligned and fired | 15 (68 bursts) |
| Intents ending in target loss or timeout | 40 |
| View commands from the connectome / overridden | 5,071 / 1,233 (80.4%) |
| Acquisition to first shot | 0.11–3.76 s, median about 1.3 s |
| Damage taken | none; all four grunts died |

![After the first grunt](results/halflife/after-first-grunt.jpg)
![After three grunts](results/halflife/after-three-grunts.jpg)

The player never took damage and the grunts died, but the numbers also show what is not yet
right: many intents timed out while the tracker chased shell casings or a target that kept
strafing, the model's share fell to about a quarter against fast-moving targets, and a burst
is fired only after settling, which costs up to three seconds. Weapon switching and reloading
by reflex worked mechanically but their colour cues (red digits, damage indicator) misfired
under the level's red emergency lighting, so they were disarmed. Traversal is still mostly the
agent's: the pilot's explore/seek behaviour bumps and turns on the change sense and stops on a
goal or motion, which got it out of the elevator but not through the level.

## Second engagement: the motion channel fed live

Same corridor, later the same day, with the adapter-v4 checkpoint (the visual slip of a turning
view trained into the lptc channel), the model in its own process with the polling wait, and a
flow watch over the upper view feeding that channel live (`core --shadow-process
--lptc-from-flow`). Another session had Liftoff loaded in the same seat; with it rendering in
the background the first attempt's capture stalled and the core halted before the fight (127
dropped observations in forty seconds). Minimising it and running the flow at eighth scale gave
a clean run: 4 dropped observations in 4,686 view
commands. Two grunts were spawned against the player in the elevator, so the fight was
point-blank ([ledger summary](results/halflife/lptc-engagement-2026-09-17.json)).

| Measurement | This session | First session |
|---|---:|---:|
| View commands | 4,686 | 6,304 |
| From the model / overridden / stale | 56.1% / 37.7% / 6.2% | 80.4% / 19.6% / not counted |
| Reflex firings, align intents that fired | 41, 8 | 58, 15 |
| Inference p50 / p95 / p99 | 4.6 / 8.8 / 10.1 ms | not in the scorecard |
| Damage taken | none | none |

Two findings. Inference stayed inside the evidence budget throughout the fight, so the
out-of-process worker and the polling wait hold up in a game, not only in the Arena. And the
live flow did not match the simulated slip: over 4,303 samples with turns above
100 px/s, the flow's translation was 0.22 of the turn the runtime
had applied 60 ms earlier (median 512 px/s against 1083),
with a correlation of 0.22. The scene explains most of it: a dark elevator
with the grunts filling the view leaves the phase correlation and the dense flow little
texture, and the flow's comparison window straddles the fast turns the align program makes
(up to 80 counts per tick). The v4 readout, trained on a slip that tracks the view exactly, was
overridden more than the first session's readout; whether that is the mismatch or the readout
cannot be told apart here. The calibration turn below settles what the percept reports.

## Calibration: what the flow percept reports for a known turn

The next day, in the same elevator with the doors open, the pilot's `calibrate` op swept the
view right then left at each of eight rates (a spread `look` of half a second, a quarter above
2,500 px/s), sampling the snapshot's flow summary about every 20 ms during and after each
sweep, with the flow watch over the upper 1280×560 at eighth and at quarter scale
([results](results/halflife/flow-calibration-2026-09-18.json)). "Applied" is the rate the
counts should produce at 1.067 counts per pixel, the figure the pilot's aim uses.

| Turn (px/s) | Reported / applied, scale 8 | Phase response, scale 8 | Reported / applied, scale 4 | Phase response, scale 4 |
|---:|---:|---:|---:|---:|
| 300 | 0.85 (0.80 to 0.89) | 0.93 | 0.92 (0.92 to 0.93) | 0.95 |
| 600 | 0.88 (0.88 to 0.88) | 0.94 | 0.93 (0.92 to 0.93) | 0.85 |
| 1,000 | 0.86 (0.86 to 0.86) | 0.88 | 0.91 (0.90 to 0.91) | 0.71 |
| 1,500 | 0.84 (0.83 to 0.85) | 0.89 | 0.87 (0.86 to 0.89) | 0.72 |
| 2,500 | 0.80 (0.79 to 0.81) | 0.75 | 0.82 (0.81 to 0.84) | 0.46 |
| 4,000 | 0.72 (0.70 to 0.75) | 0.64 | 0.75 (0.66 to 0.83) | 0.47 |
| 6,000 | 0.70 (0.69 to 0.71) | 0.22 | 0.65 (0.64 to 0.66) | 0.25 |
| 8,000 | 0.28 (0.11 to 0.45) | 0.10 | 0.43 (0.20 to 0.67) | 0.07 |

Three things follow. **Timing matches training.** The flow at a sample time is best explained
by a 60 ms box of view motion ending 20 ms before the sample (correlation
0.994), which is exactly the training world's model of the percept (a 60 ms window seen
two ticks late). Over the twenty sweeps at or below 2,500 px/s, half the response arrives 92 ms after the command
(p90 107 ms), the box's half width plus the command's own dispatch.
**Scale is the projection, not the percept.** The reported motion is 0.8 to 0.9 of "applied"
at the rates that matter, and the shortfall is in the applied figure: 1.067 counts per pixel
spreads 90° evenly over 1,280 px, but in a 90° perspective view the centre moves about 0.79
times as far per count as that average, and the percept measures the picture. No scale
correction was applied. **The range ends near 7,000 px/s.** The percept holds about 0.7 of the
turn at 4,000 to 6,000 px/s, then breaks: a third of the field over the lag is the phase
correlation's alignment limit, and the dense flow alone cannot follow such a shift.

Re-reading the fight's ledger by the rate applied in the 60 ms before each sample shows the
live flow was poor at every rate, not only the fast ones (median reported / applied 0.56 at
100 to 500 px/s, 0.45 at 1,000 to 2,000, 0.19 at 4,000 to 7,000; the best lag window on that
ledger correlates at 0.22), and it reported up to 500 px/s (90th percentile) while nothing was
applied. That is the scene: two grunts filling a point-blank view are the majority of the
picture, and their motion, not the view's, is what the fit followed. Three changes came out of
this. The flow summary now carries a `credible` flag (the alignment inside the percept's range
and at least half the sampled vectors agreeing with it), the affine fit is seeded with the
phase correlation's shift so a large mover cannot drag it, and the runtime hands the lptc
channel only a credible summary (zeros otherwise, the value the cursor episodes trained with).
The pilot's `engage` op caps the align step at 2,500 px/s of view motion (30 counts a tick at
its gain) instead of 80 counts (6,700 px/s), inside both the percept's range and the training
world's speeds.

## The channel fed against zeros: repeated engagements

Every live trial from the first fights on the 17th to 15:00 on the 18th ran with a stray
controller in the seat: a virtual Xbox pad left alive by another session's flight bridge held
its left stick fully down from 18:28 on the 16th to 15:00 on the 18th (apart from two gaps of
five and twelve minutes on the evening of the 17th), and Half-Life's config has `joystick "1"`,
so the game read a continuous move-backward through the calibration and every block above.
All conditions shared it and the comparisons between them stand; the absolute figures describe
a fight with the player pushed against the wall behind, not a free-standing one, and a bias of
this kind need not be an even handicap: a policy that wanted to back off was helped rather
than hurt. The trials below are the first without it.

The same night, with these changes, `ganglion.evaluation.halflife.trials` ran blocks of five
engagements from one quicksave in the corridor outside the elevator: quickload, two grunts
spawned from the console, a step back, the motion → track → connectome align → fire chain
armed, fourteen seconds of fight, disarm. The v4 checkpoint ran in its own process with
`--lptc-from-flow` throughout; the condition was the flow watch (eighth scale over the upper
view), present for the "fed" blocks and absent for the "zeros" blocks, so the adapter fed the
channel either the credible live summary or nothing. Blocks alternated; the ledger was sliced
per trial ([results](results/halflife/channel-trials-2026-09-18.json), per-trial files under
[results/halflife/channel-trials/](results/halflife/channel-trials/)). Acquisition is the time
from an align intent's start to its first shot's release; engagement runs to the intent's last
align step, which includes the re-alignment between shots and any tracking after the last one.

| Measurement | Channel fed (flow watch) | Channel zeros (no flow watch) |
|---|---:|---:|
| Trials | 20 | 20 |
| Align intents, of which fired | 81 of 98 (83%) | 72 of 98 (73%) |
| Acquisition to the first shot, median (p75) | 0.92 s (1.31) | 0.94 s (1.16) |
| Engagement to the last align step, median | 1.88 s | 2.13 s |
| Tracking error of align steps, median of trial means | 153 px | 143 px |
| From the model / overridden | 68.3% / 27.0% | 88.8% / 7.1% |
| Model proposals within 60° of the goal | 56% | 90% |
| Model samples carrying a credible flow | 7,742 | 0 |

Twenty trials each. With the channel fed, the chain fired on 83% of its align intents
against 73% with zeros (two-sided p 0.12) and reached its first shot in
0.92 against 0.94 s at the median (p 0.82); the tracking error of
the align steps was no better (p 0.57), and the model was overridden about four times as
often (27% of steps against 7%; per-trial model share p 2e-06). The
live channel changes what the model proposes, mostly into disagreement with the envelope, so
whatever the outcome gained is as likely the envelope's deterministic steps as the model's. In
these trials the flow itself followed the applied turn (per-trial median slope 0.50,
correlation 0.92); the 0.22 of the first engagement was the uncapped turns and the
ungated summary.

Where the proposals point says why. Over every align step the model was asked for, its
proposal's cosine with the goal error averaged +0.82 with the channel at zero
(90% of proposals within 60° of the goal, 7% pointing away) and +0.37 with it
fed (56% and 26%); and against the flow it was given, the fed proposals averaged a
cosine of -0.46, 62% of them within 60° of *opposite* to the flow. The readout turns
against the slip, which in the training world is the same as continuing its own turn: the slip
there is minus the view's own motion and nothing else, so the channel handed back the
own-velocity shortcut that adapter v3 had removed from the haltere channel. Live, the flow also
carries the grunt's motion at point blank, the two cues disagree, and the readout follows the
slip off the goal until the envelope overrides it.

A day later the flow percept learned to keep known movers out of its ego-motion: what the track
and motion watches follow is windowed out of the phase correlation and kept out of the fit.
Twenty more fed trials with that in place, after a game restart from the same quicksave
(`lptc-excl`): the flow followed the applied turn about as before (slope 0.50, correlation
0.77) and the readout behaved as before, 54% of its proposals against the flow,
46% at the goal, 35% of steps overridden, 65 of 92 (71%) intents fired (against the fed
trials above p 0.05; first shot at 0.92 s, p 0.45). The percept was not the
problem in these trials, and the readout's turning against the slip is its own: a memoryless
teacher imitated with any own-motion signal in view, in whatever channel, leaves the readout
the same shortcut, and the slip is that signal. Between-session variation (fed trials on two
days: firing 83% and 71%) is of the same size as the differences between conditions, which
bounds what twenty trials can show.

## The model against the reference, live

The same trials with the deterministic align controller (no model in the loop) and with the
v1b readout (velocity and goal error, all motor neurons, the best supervised tracker on the
suite) give the comparison the suite could only simulate. The reference and v1b blocks ran the
same night from the same quicksave, after the v4 blocks rather than alternating with them; the
v5b blocks (goal scale 0.2, the suite's settling candidate among the checkpoints that stand on
their own; v5 settles faster under supervision but was not run live) came last, after a game
restart.

| Measurement | Deterministic reference, no model | v1b readout (velocity and goal) | v5b readout (goal scale 0.2) | v4 readout, channel at zero | v4 readout, channel fed |
|---|---:|---:|---:|---:|---:|
| Trials | 20 | 20 | 20 | 20 | 20 |
| Align intents, of which fired | 82 of 113 (73%) | 69 of 89 (78%) | 56 of 93 (60%) | 72 of 98 (73%) | 81 of 98 (83%) |
| Acquisition to the first shot, median (p75) | 0.65 s (0.78) | 0.94 s (1.26) | 0.86 s (1.53) | 0.94 s (1.16) | 0.92 s (1.31) |
| Engagement to the last align step, median | 1.64 s | 1.76 s | 1.77 s | 2.13 s | 1.88 s |
| Tracking error of align steps, median of trial means | 141 px | 150 px | 158 px | 143 px | 153 px |
| From the model / overridden | n/a | 85.2% / 11.2% | 81.5% / 11.7% | 88.8% / 7.1% | 68.3% / 27.0% |
| Model proposals within 60° of the goal | n/a | 83% | 85% | 90% | 56% |
| Model samples carrying a credible flow | 0 | 0 | 0 | 0 | 7,742 |

The reference reaches its first shot sooner than the supervised v4 with the channel at zero
(0.65 against 0.94 s at the median, p 0.000) and fires the same share of its
intents (p 0.88); against v4 with the channel fed the acquisition gap is p 0.000 and the firing
gap p 0.08. v1b against v4 at zero: acquisition p 0.72, firing p 0.52; v1b against the
reference: acquisition p 0.000, firing p 0.42. v5b fired the fewest of its intents (56 of 93 (60%);
against v4 at zero firing p 0.05, acquisition p 0.56), with many late first shots (third quartile
1.53 s): the suite's settling gain did not carry into the point-blank fight.

The comparison so far was not at matched limits, though. The reference and the override step at
`max_step` counts a command (30 at the engage op's gain, 25 px, a command every 11 ms: about
2,500 px/s), while the model's accepted step is bounded by the intent speed, 1,200 px/s, and
its proposals sat at a median 11 counts; the suite gives both sides the same speed. The trials
were rerun with the turn cap at 1,200 px/s for everyone (`--max-turn-px-s 1200`), the
reference and v1b alternating block by block from the same quicksave after a game restart,
and v4 at zero after them under its own core.

| Measurement | Deterministic reference at 1,200 px/s | v1b at 1,200 px/s | v4 at zero, 1,200 px/s | v6 (goal as a direction), 1,200 px/s |
|---|---:|---:|---:|---:|
| Trials | 20 | 20 | 20 | 20 |
| Align intents, of which fired | 68 of 89 (76%) | 61 of 93 (66%) | 54 of 83 (65%) | 64 of 92 (70%) |
| Acquisition to the first shot, median (p75) | 0.97 s (1.35) | 1.24 s (1.48) | 0.99 s (1.54) | 1.07 s (1.24) |
| Engagement to the last align step, median | 1.85 s | 2.14 s | 2.02 s | 1.86 s |
| Tracking error of align steps, median of trial means | 159 px | 138 px | 151 px | 121 px |
| From the model / overridden | n/a | 80.3% / 14.3% | 87.2% / 6.2% | 76.2% / 17.8% |
| Model proposals within 60° of the goal | n/a | 80% | 92% | 78% |

At the same limit the reference still reaches its first shot sooner than v1b (0.97 against
1.24 s, p 0.01) and fires a similar share (p 0.11); the cap cost the reference
0.32 s at the median (p 0.000) and v1b 0.31 s (p 0.02), the latter through the override
steps that had been doing part of its work. v4 at zero against the reference at the same limit: first shot 0.99 against
0.97 s (p 0.30), firing p 0.10; against itself uncapped, p 0.51. The v6 readout (the goal as a direction at full strength, [TRAINING.md](TRAINING.md)),
run the same way to test the diagnosis that the model's intents time out because its proposals
fade near the goal: 64 of 92 (70%) intents fired (against the reference p 0.30, against v4 at zero
p 0.53), first shot at 1.07 s (against the reference p 0.27, against v4 at zero p 0.79),
18% of steps overridden, 78% of proposals at the goal. So the earlier gap was mostly the reference's higher step limit: at the
same limit the supervised v4 reaches its first shot as fast as the reference, with a trend
towards fewer of its intents firing, and v1b stays slower. The model does not yet make the
chain faster than the reference live, but with the limits matched it no longer makes it slower
either; the measurements to beat are on record, and fairness of the comparison is now part of
the harness (`--max-turn-px-s`, `--speed-px-s`).

## Thirty trials each, one grunt, no stray controller

With the controller gone, one grunt instead of two (fewer early losses of the tracked target),
the step limit at 1,200 px/s for both sides and blocks alternating under one core, the v6
readout and the reference were run thirty trials each (the series was cut short of forty when the user reported cursor trouble on the main desktop) ([per-trial files](results/halflife/channel-trials/),
`v6-one` and `deterministic-one`).

| Measurement (one grunt, step limit 1,200 px/s, no stray controller) | Deterministic reference | v6 readout (the goal as a direction) |
|---|---:|---:|
| Trials | 30 | 30 |
| Align intents, of which fired | 100 of 133 (75%) | 79 of 136 (58%) |
| Acquisition to the first shot, median (p75) | 0.95 s (1.30) | 1.00 s (1.46) |
| Engagement to the last align step, median | 1.67 s | 1.79 s |
| Tracking error of align steps, median of trial means | 140 px | 134 px |
| From the model / overridden | n/a | 86.1% / 11.1% |
| Model proposals within 60° of the goal | n/a | 87% |

v6 against the reference: firing p 0.003, first shot 1.00 against 0.95 s (p 0.61),
tracking error p 0.76. Against the same conditions' two-grunt trials with the controller present,
the reference's first shot moved from 0.97 to 0.95 s (p 0.75) and v6's from 1.07 to
1.00 s (p 0.65). With the bias gone and the sample doubled, the picture is plainer than before: the supervised v6 reaches its first shot as fast as the reference and tracks the target as tightly, but fires on fewer of its intents, 58% against 75%, and that gap is now significant. Its earlier tracking advantage (121 against 159 px) does not survive the controller's removal: the backward push had helped the model's condition more than the reference's, as the other session warned it might. On this fight, with the limits matched and the seat clean, the model under supervision equals the reference on speed and tracking and loses on finishing; the measurements to beat are these.

## Limits and next work

No level was completed. Cheats supplied the loadout and the test enemies; the fights are a
reflex test, not a playthrough. The perception is motion plus tracking, with no notion of
friend or foe. The next steps are a looming/approach percept for dodging, a settle-free burst
rule for close targets, weapon/ammo cues taught as templates rather than colours, and a
traversal behaviour with a real sense of open space.

## Reproduce

Inside the seat tool with the game running and Steam launched through the seat:

```powershell
.venv/Scripts/python.exe -m ganglion.cli core --pid <game pid> --seconds 1750 --endpoint runs/hl.endpoint.json --shadow-checkpoint runs/cursor-dagger-v1/round-03/cursor-readout.pt
.venv/Scripts/python.exe -m ganglion.evaluation.halflife.pilot serve --endpoint runs/hl.endpoint.json --dir runs/hl/pilot --session 2
.venv/Scripts/python.exe -m ganglion.evaluation.halflife.pilot do --dir runs/hl/pilot "{\"op\":\"engage\",\"response\":\"track\",\"controller\":\"connectome\"}"
.venv/Scripts/python.exe -m ganglion.evaluation.halflife.pilot do --dir runs/hl/pilot "{\"op\":\"watch\",\"name\":\"flow\",\"kind\":\"flow\",\"region\":[0,0,1280,560],\"flow_scale\":8}"
.venv/Scripts/python.exe -m ganglion.evaluation.halflife.pilot do --dir runs/hl/pilot --timeout 90 "{\"op\":\"calibrate\",\"seconds\":0.5}"
.venv/Scripts/python.exe -m ganglion.evaluation.halflife.trials run --dir runs/hl/pilot --label lptc-on-1 --trials 5 --flow-scale 8
.venv/Scripts/python.exe -m ganglion.evaluation.halflife.trials run --dir runs/hl/pilot --label channel-zero-1 --trials 5
.venv/Scripts/python.exe -m ganglion.evaluation.halflife.trials report runs/hl/trials/lptc-on-*.json runs/hl/trials/channel-zero-*.json
```

The trials need a quicksave (F6) at the fight's start with the loadout given and god mode on; the
per-trial files behind the comparison are in [results/halflife/channel-trials/](results/halflife/channel-trials/).

The calibration needs the game in the foreground when the core starts (it halts on a target
that is not the foreground window) and a scene with texture; `r_fullbright 1` did nothing in
this build.

`valve/ganglion.cfg` in the game folder sets raw input, fast weapon switching and a `level`
alias bound to End. The grave key must be sent by scan code on non-US layouts.
