"""
test_environment.py
Bộ kiểm thử cho caro_environment.py.

Chạy tại thư mục chứa hai file:
    python test_environment.py
"""

from caro_environment import CaroEnv, EMPTY, X, O


def test_reset_and_initial_state():
    env = CaroEnv()
    observation = env.reset()

    assert len(observation) == 11
    assert all(len(row) == 11 for row in observation)
    assert all(cell == EMPTY for row in observation for cell in row)
    assert env.current_player == X
    assert env.winner == EMPTY
    assert env.done is False
    assert env.last_move is None
    assert len(env.get_legal_moves()) == 121
    print("PASS: Khởi tạo và reset bàn cờ 11x11")


def test_valid_moves_and_turn_switch():
    env = CaroEnv()

    _, reward, terminated, truncated, info = env.step((5, 5))
    assert reward == 0
    assert terminated is False
    assert truncated is False
    assert info["player"] == X
    assert env.board[5][5] == X
    assert env.current_player == O

    _, reward, terminated, _, info = env.step((0, 0))
    assert reward == 0
    assert terminated is False
    assert info["player"] == O
    assert env.board[0][0] == O
    assert env.current_player == X
    print("PASS: Nước đi hợp lệ và đổi lượt X/O")


def test_invalid_actions_do_not_change_state():
    invalid_actions = [
        None,
        "5,5",
        (5,),
        (5, 5, 1),
        (-1, 0),
        (0, -1),
        (11, 0),
        (0, 11),
        (1.5, 2),
        (True, 2),
    ]

    for action in invalid_actions:
        env = CaroEnv()
        before_board = env.get_observation()
        before_player = env.current_player
        before_done = env.done

        _, reward, terminated, truncated, info = env.step(action)

        assert reward == -1
        assert terminated is False
        assert truncated is False
        assert "error" in info
        assert env.board == before_board
        assert env.current_player == before_player
        assert env.done == before_done

    # Đánh vào ô đã có quân cũng phải bị từ chối, không đổi lượt.
    env = CaroEnv()
    env.step((2, 2))
    player_before = env.current_player
    board_before = env.get_observation()

    _, reward, terminated, _, info = env.step((2, 2))

    assert reward == -1
    assert terminated is False
    assert "error" in info
    assert env.board == board_before
    assert env.current_player == player_before
    print("PASS: Tọa độ sai và ô đã đánh không làm thay đổi trạng thái")


def _prepare_winning_position(direction, player=X, run_length=4):
    """Tạo bàn cờ có run_length quân liên tiếp để kiểm tra nước thắng kế tiếp."""
    env = CaroEnv()
    env.current_player = player

    if direction == "horizontal":
        start = (5, 2)
        dr, dc = 0, 1
    elif direction == "vertical":
        start = (2, 5)
        dr, dc = 1, 0
    elif direction == "diagonal":
        start = (2, 2)
        dr, dc = 1, 1
    elif direction == "anti_diagonal":
        start = (2, 8)
        dr, dc = 1, -1
    else:
        raise ValueError("Hướng kiểm thử không hợp lệ.")

    for k in range(run_length):
        env.board[start[0] + k * dr][start[1] + k * dc] = player

    winning_move = (
        start[0] + run_length * dr,
        start[1] + run_length * dc,
    )
    return env, winning_move


def test_win_in_four_directions():
    directions = (
        "horizontal",
        "vertical",
        "diagonal",
        "anti_diagonal",
    )

    for direction in directions:
        env, winning_move = _prepare_winning_position(direction, X)
        _, reward, terminated, truncated, info = env.step(winning_move)

        assert reward == 1, direction
        assert terminated is True, direction
        assert truncated is False
        assert info["winner"] == X, direction
        assert env.done is True

    print("PASS: X thắng theo cả 4 hướng ngang, dọc, chéo chính, chéo phụ")


def test_o_can_win():
    env, winning_move = _prepare_winning_position("horizontal", O)
    _, reward, terminated, _, info = env.step(winning_move)

    assert reward == 1
    assert terminated is True
    assert info["winner"] == O
    print("PASS: O cũng có thể chiến thắng")


def test_six_or_more_consecutive_marks():
    # Đã có 5 quân X; đánh quân thứ 6 vẫn phải được tính là thắng.
    env = CaroEnv()
    env.current_player = X

    for col in range(2, 7):
        env.board[5][col] = X

    _, reward, terminated, _, info = env.step((5, 7))

    assert reward == 1
    assert terminated is True
    assert info["winner"] == X
    print("PASS: 6 quân liên tiếp vẫn được công nhận chiến thắng")


def test_reset_after_game():
    env, winning_move = _prepare_winning_position("horizontal", X)
    env.step(winning_move)
    assert env.done is True
    assert env.winner == X

    observation = env.reset()

    assert all(cell == EMPTY for row in observation for cell in row)
    assert env.current_player == X
    assert env.winner == EMPTY
    assert env.done is False
    assert env.last_move is None
    assert len(env.get_legal_moves()) == 121
    print("PASS: Reset sau khi ván kết thúc")


def test_no_moves_after_game_over():
    env, winning_move = _prepare_winning_position("horizontal", X)
    env.step(winning_move)

    board_before = env.get_observation()
    player_before = env.current_player

    _, reward, terminated, truncated, info = env.step((0, 0))

    assert reward == -1
    assert terminated is True
    assert truncated is False
    assert "error" in info
    assert env.board == board_before
    assert env.current_player == player_before
    print("PASS: Không thể tiếp tục đánh sau khi ván kết thúc")


def test_draw_detection():
    env = CaroEnv()

    # Tạo bàn cờ kín theo mẫu tuần hoàn.
    # Mẫu này không tạo 5 quân liên tiếp theo 4 hướng thắng.
    for row in range(env.board_size):
        for col in range(env.board_size):
            value = (row + col + row // 2) % 2
            env.board[row][col] = X if value == 0 else O

    assert len(env.get_legal_moves()) == 0
    assert env.check_winner() == EMPTY
    env.winner = EMPTY
    assert env.is_draw() is True

    # Đồng bộ trạng thái kết thúc như sau một ván hòa.
    env.done = True
    assert env.is_draw() is True
    print("PASS: Nhận diện bàn cờ đầy và không có người thắng (hòa)")


def run_all_tests():
    tests = [
        test_reset_and_initial_state,
        test_valid_moves_and_turn_switch,
        test_invalid_actions_do_not_change_state,
        test_win_in_four_directions,
        test_o_can_win,
        test_six_or_more_consecutive_marks,
        test_reset_after_game,
        test_no_moves_after_game_over,
        test_draw_detection,
    ]

    print("=" * 60)
    print("BẮT ĐẦU KIỂM THỬ CORE ENVIRONMENT - CARO 11x11")
    print("=" * 60)

    for test in tests:
        test()

    print("=" * 60)
    print(f"TẤT CẢ {len(tests)} NHÓM KIỂM THỬ ĐỀU PASS!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
