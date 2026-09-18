# Preserve main Steam while launching a game in the seat

Observed 2026-09-17 with The Zachtronics Solitaire Collection, app 1988540, build 24998607.

The desired behavior is a usable Steam client on the main desktop while the game runs and
receives input inside the seat. The seat isolates input/capture correctly, but the observed
Steam startup path did not preserve that client arrangement.

1. Main Steam initially ran in session 1. A direct `seat_run` of the installed game started in
   session 3, then exited and requested `steam://run/1988540`.
2. Steam's bootstrap log recorded a shutdown followed by a session-3 client startup. The game
   relaunched in session 3 and passed the Ganglion card-move checks.
3. A surviving main-session `steam.exe` was a pending launch command, with no main-session UI
   helpers. Counting Steam processes alone would have incorrectly suggested coexistence.
4. After closing the test game/client and restoring main Steam, a temporary app-ID file prevented
   the redirect but the game showed “You must start Steam before launching the game.” Main Steam
   remained running. The file was removed and the error dialog dismissed.

See the [scorecard and cleanup record](../docs/bench/SOLITAIRE.md). No authentication settings,
game binaries, or permanent launch configuration were changed.

Suggested next investigation, confined to the seat tool's launcher and diagnostics:

- Distinguish launch requests, a usable Steam client/UI, and the actual game session. Verify
  the result after startup; a successful `seat_run` response only locates the initial process.
- Report a game-initiated Steam relaunch and parent-client shutdown explicitly. Do not claim
  unchanged main-desktop Steam access just because its executable still appears in process lists.
- Investigate supported Steam client IPC/session constraints before offering a concurrent mode.
  Keep the existing conflict protection until coexistence is demonstrated. For this game,
  an app-ID file alone was insufficient.
- Define a coexistence test with a usable main client before and after launch, a responsive seat
  game, seat-only input, and cleanup that preserves the main client's session and window.

This is a measured launch limitation separate from Ganglion's successful input/perception transfer.
