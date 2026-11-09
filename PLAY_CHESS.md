# Play Midnight chess

From this folder, run:

```powershell
python -m pip install -r requirements.txt
python chess_game.py
```

The existing project environment can also be used directly:

```powershell
.\chess\Scripts\python.exe chess_game.py
```

You play White. Enter moves in the terminal (`e4`, `Nf3`, `e2e4`,
`O-O`, or `e7e8q` for promotion), or use the From/To sliders in
the simulation and press **Make my move**. Files use 0=a through 7=h;
ranks use 1 through 8. Enter `new` to restart or `quit` to leave.

Use `python chess_game.py --robot-color white` to play Black.
Stockfish is bundled; `--skill 0` through `--skill 20` sets its strength.
Only reachable moves use the arm. Other moves and failed grasps are
completed by simulated hand movement so play can continue.

Verification:

```powershell
python test_chess_game.py
python chess_game.py --direct --plies 12 --movetime 0.05
```

The integration suite covers special moves, failed arm capture recovery,
new games, illegal moves, checkmate, and a complete Stockfish game.
The second command also exercises actual arm physics without opening a window.
