"""Play chess against Midnight.

You move the white pieces (type the move in this terminal, or use the sliders + "Make my move"
button in the PyBullet window); Midnight answers with its arm, deciding its moves with a real
chess engine (Stockfish, driven through python-chess over the UCI protocol).

    python chess_game.py                         # you = white, Midnight = black
    python chess_game.py --robot-color white     # Midnight moves first
    python chess_game.py --skill 5 --movetime 1  # engine strength (0..20) / thinking time (s)
    python chess_game.py --engine PATH           # a different UCI engine
    python chess_game.py --stub                  # no engine: random legal moves (for testing)
    python chess_game.py --direct --stub         # headless self-test: both sides play by themselves

Midnight's arm can only reach part of the board (see the map printed at start-up). It therefore
only chooses among the legal moves it can physically carry out (no castling or promotion; captures
are done by first moving the captured piece to a slot beside the board). If it has no such move
it says so and makes that move "by hand" (teleported) so the game can go on.

Typing a move: "e2e4", "e4", "Nf3", "O-O" ... ("new" = new game, "quit" = leave).
"""
import argparse
import math
import os
import queue
import random
import shutil
import sys
import threading
import time

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENGINE = os.path.join(HERE, "engines", "stockfish", "stockfish",
                              "stockfish-windows-x86-64-universal.exe")

ap = argparse.ArgumentParser(description="Play chess against Midnight.", add_help=True)
ap.add_argument("--engine", help="path to a UCI engine (default: bundled Stockfish, else 'stockfish' on PATH)")
ap.add_argument("--skill", type=int, default=6, help="Stockfish 'Skill Level' 0..20 (default 6)")
ap.add_argument("--movetime", type=float, default=1.0, help="engine thinking time per move in seconds")
ap.add_argument("--robot-color", choices=("black", "white"), default="black")
ap.add_argument("--stub", action="store_true", help="random legal moves instead of an engine (testing)")
ap.add_argument("--direct", action="store_true", help="headless; both sides play automatically")
ap.add_argument("--plies", type=int, default=12, help="with --direct: how many half-moves to play")
ap.add_argument("--seed", type=int, default=1)
args, _ = ap.parse_known_args()

import chess_robot_sim as sim            # builds the physics scene (reads --direct from sys.argv)
from chess_robot_sim import p            # noqa: E402  (PyBullet)

ROBOT_WHITE = args.robot_color == "white"
ROBOT_COLOR = chess.WHITE if ROBOT_WHITE else chess.BLACK
random.seed(args.seed)


# ----------------------------------------------------------------------------------------
# Engines
# ----------------------------------------------------------------------------------------
class RandomEngine:
    name = "random mover (stub)"

    def choose(self, board, root_moves=None):
        return random.choice(list(root_moves) if root_moves else list(board.legal_moves))

    def close(self):
        pass


class UciEngine:
    def __init__(self, path, skill, movetime):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.movetime = movetime
        self.name = self.engine.id.get("name", os.path.basename(path))
        options = self.engine.options
        cfg = {}
        if "Skill Level" in options:
            cfg["Skill Level"] = max(0, min(20, skill))
        if "Threads" in options:
            cfg["Threads"] = 2
        if "Hash" in options:
            cfg["Hash"] = 64
        if cfg:
            self.engine.configure(cfg)

    def choose(self, board, root_moves=None):
        res = self.engine.play(board, chess.engine.Limit(time=self.movetime),
                               root_moves=list(root_moves) if root_moves else None)
        return res.move

    def close(self):
        self.engine.quit()


def make_engine():
    if args.stub:
        return RandomEngine()
    path = args.engine or (DEFAULT_ENGINE if os.path.exists(DEFAULT_ENGINE) else shutil.which("stockfish"))
    if not path or not os.path.exists(path):
        sys.exit("No chess engine found. Pass --engine PATH (any UCI engine, e.g. Stockfish) "
                 "or use --stub for random moves.")
    return UciEngine(path, args.skill, args.movetime)


# ----------------------------------------------------------------------------------------
# Board <-> simulation bookkeeping
# ----------------------------------------------------------------------------------------
board = chess.Board()
yard_free = {"R": [f"R{i}" for i in range(5)], "L": [f"L{i}" for i in range(5)]}
upright = {b: orn for b, (_, (_, orn)) in sim.start_pose.items()}   # resting orientation per piece
promoted_bodies = []
status_id = None


def status(text, console=True):
    global status_id
    if console:
        print(text, flush=True)
    if not sim.DIRECT:
        kw = dict(textPosition=[-0.17, -0.355, 0.03], textColorRGB=[0.05, 0.25, 0.05], textSize=1.5)
        status_id = p.addUserDebugText(text, replaceItemUniqueId=status_id, **kw) if status_id is not None \
            else p.addUserDebugText(text, **kw)


def sq_name(s):
    return chess.square_name(s)


def resting_com_z():
    return sim.board_top + sim.PIECE_COM_HEIGHT + 0.0002


def snap_to(body, name):
    """Put a piece upright and exactly on the centre of square/slot `name`."""
    x, y = sim.square_xy_cm(name)
    p.resetBasePositionAndOrientation(body, [x / 100, y / 100, resting_com_z()], upright[body])
    p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])


def resync():
    """After a robot move: any piece that is not where the game thinks it is gets put back."""
    fixed = []
    for name, body in sim.piece_at.items():
        (px, py, pz), _ = p.getBasePositionAndOrientation(body)
        x, y = sim.square_xy_cm(name)
        err_mm = math.hypot(px * 100 - x, py * 100 - y) * 10
        if err_mm > 9.0 or pz < sim.board_top:
            snap_to(body, name)
            fixed.append(f"{name} (was {err_mm:.0f} mm off)")
    if fixed:
        print("  tidied up:", ", ".join(fixed))
    sim.wait(0.3)


def hand_move(body, name, seconds=1.2):
    """A 'hand' (teleport, no arm) carries a piece to square/slot `name`: used for the human's moves."""
    x, y = sim.square_xy_cm(name)
    (sx, sy, sz), _ = p.getBasePositionAndOrientation(body)
    lift = 0.05
    n = max(1, int(seconds / sim.DT))
    for k in range(1, n + 1):
        u = k / n
        s = u * u * (3 - 2 * u)
        z = sz + (resting_com_z() - sz) * s + lift * math.sin(math.pi * u)
        p.resetBasePositionAndOrientation(body, [sx + (x / 100 - sx) * s, sy + (y / 100 - sy) * s, z], upright[body])
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        sim.drive(sim.cur)
        sim._tick()
    snap_to(body, name)


def park_captured(square, side):
    """Take the piece standing on `square` off the board into a slot of `side` ('R' / 'L')."""
    body = sim.piece_at.pop(square)
    if yard_free[side]:
        slot = yard_free[side].pop(0)
        hand_move(body, slot)
        sim.piece_at[slot] = body
    else:                                        # no slot left: just take it off the table
        p.resetBasePositionAndOrientation(body, [0.8, 0.8, 0.1], upright[body])
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
    return body


def verify_sync():
    """Check that the simulated pieces match the python-chess board (type and position)."""
    bad = []
    for s in chess.SQUARES:
        piece = board.piece_at(s)
        name = sq_name(s)
        if piece is None:
            if name in sim.piece_at:
                bad.append(f"{name}: sim has a piece, board is empty")
            continue
        body = sim.piece_at.get(name)
        if body is None:
            bad.append(f"{name}: board has {piece.symbol()}, sim is empty")
        elif sim.piece_kind[body] != piece.symbol().upper():
            bad.append(f"{name}: board {piece.symbol()} vs sim {sim.piece_kind[body]}")
        else:
            (px, py, pz), _ = p.getBasePositionAndOrientation(body)
            x, y = sim.square_xy_cm(name)
            if math.hypot(px * 100 - x, py * 100 - y) > 1.0:
                bad.append(f"{name}: piece is {math.hypot(px * 100 - x, py * 100 - y) * 10:.0f} mm off")
    return bad


# ----------------------------------------------------------------------------------------
# Moves
# ----------------------------------------------------------------------------------------
def captured_square(move):
    if board.is_en_passant(move):
        return sq_name(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
    if board.is_capture(move):
        return sq_name(move.to_square)
    return None


def robot_can_do(move):
    """Can Midnight's arm carry out this move?"""
    if board.is_castling(move) or move.promotion:
        return False
    frm, to = sq_name(move.from_square), sq_name(move.to_square)
    if frm not in sim.REACHABLE or to not in sim.REACHABLE:
        return False
    cap = captured_square(move)
    if cap is not None and (cap not in sim.REACHABLE or not yard_free["R"]):
        return False
    return True


def robot_execute(move):
    """Carry the move out with the arm (captured piece first goes to a slot beside the board)."""
    frm, to = sq_name(move.from_square), sq_name(move.to_square)
    cap = captured_square(move)
    ok = True
    if cap is not None:
        slot = yard_free["R"].pop(0)
        ok &= bool(sim.pick_and_place(cap, slot))
    ok &= bool(sim.pick_and_place(frm, to))
    resync()
    return ok


def apply_move_by_hand(move, side):
    """Move pieces without the arm (human moves, or robot moves it cannot reach)."""
    frm, to = sq_name(move.from_square), sq_name(move.to_square)
    cap = captured_square(move)
    if cap is not None:
        park_captured(cap, side)
    body = sim.piece_at.pop(frm)
    hand_move(body, to)
    sim.piece_at[to] = body
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = board.is_kingside_castling(move)
        r_from = sq_name(chess.square(7 if kingside else 0, rank))
        r_to = sq_name(chess.square(5 if kingside else 3, rank))
        rook = sim.piece_at.pop(r_from)
        hand_move(rook, r_to)
        sim.piece_at[r_to] = rook
    if move.promotion:                                    # swap the pawn for the new piece
        pawn = sim.piece_at.pop(to)
        sim.piece_at[to] = pawn
        park_captured(to, side)                          # pawn leaves the board
        kind = chess.piece_symbol(move.promotion).upper()
        x, y = sim.square_xy_cm(to)
        new = sim.make_piece(kind, sim.WHITE if board.turn == chess.WHITE else sim.BLACK,
                             x / 100, y / 100, sim.board_top, 0.0 if board.turn == chess.WHITE else math.pi)
        _, orn = p.getBasePositionAndOrientation(new)
        upright[new] = orn
        sim.piece_kind[new] = kind
        sim.piece_at[to] = new
        promoted_bodies.append(new)


def describe_outcome():
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        return f"Checkmate - {winner} wins ({'you' if (winner == 'White') != ROBOT_WHITE else 'Midnight'})!"
    if board.is_stalemate():
        return "Stalemate - draw."
    if board.is_insufficient_material():
        return "Draw (insufficient material)."
    if board.is_game_over():
        return f"Game over: {board.result()} ({board.outcome().termination.name.lower().replace('_', ' ')})."
    return None


def robot_turn(engine):
    status("Midnight is thinking...")
    doable = [m for m in board.legal_moves if robot_can_do(m)]
    t0 = time.time()
    if doable:
        move = engine.choose(board, root_moves=doable)
        by_arm = True
    else:
        move = engine.choose(board)
        by_arm = False
    san = board.san(move)
    note = "" if by_arm else "  (out of the arm's reach - moved by hand)"
    status(f"Midnight plays {san}{note}")
    print(f"  ({len(doable)} of {board.legal_moves.count()} legal moves were within the arm's reach; "
          f"engine took {time.time() - t0:.1f} s)")
    if by_arm:
        if not robot_execute(move):
            print("  the move did not go perfectly - the board was tidied up")
    else:
        apply_move_by_hand(move, "R")
    board.push(move)


def human_turn_apply(move):
    san = board.san(move)
    apply_move_by_hand(move, "L")
    board.push(move)
    status(f"You played {san}")


# ----------------------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------------------
typed = queue.Queue()


def _stdin_reader():
    for line in sys.stdin:
        typed.put(line.strip())


def parse_typed(text):
    try:
        return board.parse_san(text)
    except ValueError:
        pass
    try:
        mv = board.parse_uci(text.lower())
        return mv
    except ValueError:
        return None


def read_move_from_ui(ui):
    f1, r1, f2, r2 = (int(round(p.readUserDebugParameter(ui[k]))) for k in ("f1", "r1", "f2", "r2"))
    frm = chess.square(f1, r1 - 1)
    to = chess.square(f2, r2 - 1)
    promo = {1: chess.QUEEN, 2: chess.ROOK, 3: chess.BISHOP, 4: chess.KNIGHT}[
        int(round(p.readUserDebugParameter(ui["promo"])))]
    mv = chess.Move(frm, to)
    piece = board.piece_at(frm)
    if piece and piece.piece_type == chess.PAWN and chess.square_rank(to) in (0, 7):
        mv = chess.Move(frm, to, promotion=promo)
    return mv


def new_game():
    global board, status_id
    board = chess.Board()
    for b in promoted_bodies:
        p.resetBasePositionAndOrientation(b, [0.8, 0.8, 0.1], [0, 0, 0, 1])
    promoted_bodies.clear()
    sim.reset_board()
    yard_free["R"] = [f"R{i}" for i in range(5)]
    yard_free["L"] = [f"L{i}" for i in range(5)]
    status("New game. " + ("Midnight moves first." if ROBOT_WHITE else "Your move (white)."))


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def run_direct(engine):
    """Headless self-test: both sides play automatically; Midnight's side uses the arm."""
    print(f"\nHeadless game, {args.plies} half-moves; engine: {engine.name}\n")
    arm_moves = hand_moves = 0
    for ply in range(args.plies):
        if board.is_game_over():
            break
        if board.turn == ROBOT_COLOR:
            doable = [m for m in board.legal_moves if robot_can_do(m)]
            arm_moves += bool(doable)
            hand_moves += not doable
            robot_turn(engine)
        else:
            mv = engine.choose(board)
            print(f"\nhuman side plays {board.san(mv)}")
            human_turn_apply(mv)
        bad = verify_sync()
        print("  sync check:", "OK" if not bad else bad)
        if bad:
            break
    print(f"\nmoves by the arm: {arm_moves}, moves by hand: {hand_moves}")
    print(board)
    out = describe_outcome()
    if out:
        print(out)


def run_gui(engine):
    threading.Thread(target=_stdin_reader, daemon=True).start()
    p.resetDebugVisualizerCamera(0.95, 15, -38, [0, -0.08, 0.12])
    ui = {
        "f1": p.addUserDebugParameter("From file (0=a..7=h)", 0, 7, 4),
        "r1": p.addUserDebugParameter("From rank (1..8)", 1, 8, 2),
        "f2": p.addUserDebugParameter("To file (0=a..7=h)", 0, 7, 4),
        "r2": p.addUserDebugParameter("To rank (1..8)", 1, 8, 4),
        "promo": p.addUserDebugParameter("Promote to (1=Q 2=R 3=B 4=N)", 1, 4, 1),
        "go": p.addUserDebugParameter("Make my move", 1, 0, 0),
        "new": p.addUserDebugParameter("New game", 1, 0, 0),
    }
    last = {"go": p.readUserDebugParameter(ui["go"]), "new": p.readUserDebugParameter(ui["new"])}
    status(f"Engine: {engine.name}. " + ("Midnight moves first." if ROBOT_WHITE else
                                          "Your move (white): type e.g. e2e4 here, or use the sliders."))
    announced = False
    while p.isConnected():
        if p.readUserDebugParameter(ui["new"]) != last["new"]:
            last["new"] = p.readUserDebugParameter(ui["new"])
            new_game(); announced = False
            continue
        if board.is_game_over():
            if not announced:
                status(describe_outcome() + "  (press 'New game')")
                announced = True
        elif board.turn == ROBOT_COLOR:
            robot_turn(engine)
            out = describe_outcome()
            if out:
                continue
            status(f"Midnight played. Your move{' - you are in check!' if board.is_check() else '.'}")
        else:
            move, text = None, None
            if p.readUserDebugParameter(ui["go"]) != last["go"]:
                last["go"] = p.readUserDebugParameter(ui["go"])
                move, text = read_move_from_ui(ui), "that move"
            else:
                try:
                    text = typed.get_nowait()
                except queue.Empty:
                    text = None
                if text:
                    if text.lower() in ("quit", "exit", "q"):
                        break
                    if text.lower() == "new":
                        new_game(); announced = False
                        continue
                    move = parse_typed(text)
            if move is not None or text:
                if move is None or move not in board.legal_moves:
                    status(f"'{text}' is not a legal move - try again.")
                else:
                    human_turn_apply(move)
                    if board.is_game_over():
                        continue
        sim.drive(sim.cur)
        sim._tick()


if __name__ == "__main__":
    engine = make_engine()
    try:
        sim.print_reach_map()
        sim.settle()
        if ROBOT_WHITE and not sim.DIRECT:
            pass
        if sim.DIRECT:
            run_direct(engine)
        else:
            run_gui(engine)
    finally:
        engine.close()
        if p.isConnected():
            p.disconnect()
