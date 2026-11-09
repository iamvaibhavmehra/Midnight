"""Integration checks using the real scene and bundled Stockfish.

Run: python test_chess_game.py
"""
import sys
import unittest
from unittest.mock import patch

sys.argv = [sys.argv[0], "--direct"]
import chess_game as game


class ChessIntegrationTests(unittest.TestCase):
    def setUp(self):
        game.new_game()

    def assertSynced(self):
        self.assertEqual(game.verify_sync(), [])

    def play(self, *moves):
        for text in moves:
            move = game.parse_typed(text)
            self.assertIsNotNone(move, text)
            game.human_turn_apply(move)
            self.assertSynced()

    def test_castling(self):
        self.play("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O")
        self.assertIn("f1", game.sim.piece_at)

    def test_en_passant(self):
        self.play("e4", "a6", "e5", "d5", "exd6")
        self.assertNotIn("d5", game.sim.piece_at)

    def test_promotion_and_reset(self):
        self.play("a4", "h5", "a5", "h4", "a6", "h3", "axb7", "hxg2", "bxa8=N")
        self.assertEqual(game.sim.piece_kind[game.sim.piece_at["a8"]], "N")
        game.new_game()
        self.assertSynced()
        self.assertEqual(len(game.sim.piece_at), 32)
        self.assertEqual(len(game.sim.piece_kind), 32)

    def test_failed_arm_capture_recovers(self):
        self.play("e4", "d5")
        move = game.board.parse_san("exd5")
        # First pickup succeeds, second fails: exercise partial capture recovery.
        def pickup(src, dst):
            if src == "d5":
                body = game.sim.piece_at.pop(src)
                game.sim.piece_at[dst] = body
                game.snap_to(body, dst)
                return True
            return False
        with patch.object(game.sim, "pick_and_place", side_effect=pickup):
            self.assertFalse(game.robot_execute(move))
        game.board.push(move)
        self.assertSynced()
        self.assertEqual(len(game.yard_free["R"]), 4)

    def test_illegal_and_checkmate(self):
        self.assertIsNone(game.parse_typed("e2e5"))
        self.play("f3", "e5", "g4", "Qh4#")
        self.assertTrue(game.board.is_checkmate())
        self.assertIn("Black wins", game.describe_outcome())

    def test_stockfish_complete_game(self):
        engine = game.make_engine()
        engine.movetime = 0.01
        # Special-move tests above exercise animated movement. Run the long
        # engine game with immediate placement to keep this check practical.
        placement = patch.object(game, "hand_move", side_effect=lambda body, name: game.snap_to(body, name))
        placement.start()
        try:
            for _ in range(600):
                if game.board.is_game_over():
                    break
                move = engine.choose(game.board)
                self.assertIn(move, game.board.legal_moves)
                game.human_turn_apply(move)
                self.assertSynced()
            self.assertTrue(game.board.is_game_over(), game.board.fen())
            print("Stockfish game result:", game.board.result(), game.board.outcome())
        finally:
            placement.stop()
            engine.close()


if __name__ == "__main__":
    try:
        unittest.main(argv=[sys.argv[0]])
    finally:
        if game.p.isConnected():
            game.p.disconnect()
