# Half-Life through the reflex layer

First live first-person test, 2026-09-17, inside the Anode seat at 1280×720 (Half-Life, Steam
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

## Limits and next work

No level was completed. Cheats supplied the loadout and the test enemies; the fights are a
reflex test, not a playthrough. The perception is motion plus tracking, with no notion of
friend or foe. The next steps are a looming/approach percept for dodging, a settle-free burst
rule for close targets, weapon/ammo cues taught as templates rather than colours, and a
traversal behaviour with a real sense of open space.

## Reproduce

Inside Anode with the game running and Steam launched through the seat:

```powershell
.venv/Scripts/python.exe -m ganglion.cli core --pid <game pid> --seconds 1750 --endpoint runs/hl.endpoint.json --shadow-checkpoint runs/cursor-dagger-v1/round-03/cursor-readout.pt
.venv/Scripts/python.exe -m ganglion.evaluation.halflife.pilot serve --endpoint runs/hl.endpoint.json --dir runs/hl/pilot --session 2
.venv/Scripts/python.exe -m ganglion.evaluation.halflife.pilot do --dir runs/hl/pilot "{\"op\":\"engage\",\"response\":\"track\",\"controller\":\"connectome\"}"
```

`valve/ganglion.cfg` in the game folder sets raw input, fast weapon switching and a `level`
alias bound to End. The grave key must be sent by scan code on non-US layouts.
