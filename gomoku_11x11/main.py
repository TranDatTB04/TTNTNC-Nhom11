import sys
import pygame

from environment.caro_environment import CaroEnv, X, O
from environment.gui import (
    CELL_SIZE,
    MARGIN,
    WINDOW_SIZE,
    BLUE,
    RED,
    draw_board,
    draw_stones,
)
from agents.mcts_agent import MCTSAgent


def run_game(mode="pve", simulations=800, human_side=X):
    """
    Khởi chạy vòng lặp game Pygame.
    - mode: "pve" (Người vs AI) hoặc "pvp" (2 Người chơi)
    - simulations: Số lượt mô phỏng MCTS cho mỗi nước đi của AI
    - human_side: Quân của người chơi trong chế độ PvE (X đi trước, O đi sau)
    """
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    title = (
        f"Caro 11x11 - Người vs MCTS AI ({simulations} sims)"
        if mode == "pve"
        else "Caro 11x11 - Chế độ 2 Người chơi"
    )
    pygame.display.set_caption(title)

    font = pygame.font.SysFont("tahoma", 32, bold=True)
    status_font = pygame.font.SysFont("tahoma", 22)

    env = CaroEnv()
    board = env.reset()
    ai_agent = MCTSAgent(simulations=simulations) if mode == "pve" else None
    ai_side = -human_side

    while True:
        draw_board(screen)
        draw_stones(screen, board, env.last_move)

        # 1. Hiển thị thanh trạng thái hoặc thông báo kết quả
        if not env.done:
            if mode == "pve":
                if env.current_player == human_side:
                    side_name = "X" if human_side == X else "O"
                    status_text = f"Lượt của bạn ({side_name}) - Click vào ô trống"
                    status_color = BLUE if human_side == X else RED
                else:
                    side_name = "O" if ai_side == O else "X"
                    status_text = f"AI ({side_name}) đang suy nghĩ..."
                    status_color = RED if ai_side == O else BLUE
            else:
                status_text = (
                    "Lượt hiện tại: Người chơi X (Xanh)"
                    if env.current_player == X
                    else "Lượt hiện tại: Người chơi O (Đỏ)"
                )
                status_color = BLUE if env.current_player == X else RED

            status_surf = status_font.render(status_text, True, status_color)
            screen.blit(status_surf, (MARGIN, 10))
        else:
            if env.winner == X:
                end_text = "Người chơi X thắng! (Nhấn R / Click để chơi lại)"
            elif env.winner == O:
                end_text = "Người chơi O thắng! (Nhấn R / Click để chơi lại)"
            else:
                end_text = "Hòa cờ! (Nhấn R / Click để chơi lại)"

            text_surf = font.render(end_text, True, (200, 0, 0))
            bg_rect = text_surf.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
            pygame.draw.rect(screen, (255, 255, 255), bg_rect.inflate(24, 14))
            pygame.draw.rect(screen, (0, 0, 0), bg_rect.inflate(24, 14), 2)
            screen.blit(text_surf, bg_rect)

        pygame.display.flip()

        # 2. Lượt của AI (chỉ chạy trong chế độ PvE)
        if mode == "pve" and not env.done and env.current_player == ai_side:
            # Xử lý sự kiện hệ thống để cửa sổ Windows không bị treo ("Not Responding") lúc AI đang tính toán
            pygame.event.pump()

            action = ai_agent.get_action(env)
            if action is not None:
                next_board, reward, _, _, _ = env.step(action)
                if reward != -1:
                    board = next_board
            continue

        # 3. Xử lý sự kiện bàn phím và chuột
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Nhấn R hoặc Space để reset bàn cờ
            if event.type == pygame.KEYDOWN:
                if (
                    event.key in (pygame.K_r, pygame.K_SPACE)
                    or event.unicode.lower() == "r"
                ):
                    board = env.reset()

            # Xử lý click chuột
            if event.type == pygame.MOUSEBUTTONDOWN:
                if env.done:
                    board = env.reset()
                else:
                    # Trong chế độ PvE, chỉ nhận click khi đang tới lượt người chơi
                    if mode == "pve" and env.current_player != human_side:
                        continue

                    mouse_x, mouse_y = event.pos
                    col = (mouse_x - MARGIN) // CELL_SIZE
                    row = (mouse_y - MARGIN) // CELL_SIZE

                    next_board, reward, _, _, _ = env.step((row, col))
                    if reward != -1:
                        board = next_board


def main():
    print("=" * 50)
    print("CỜ CARO 11x11 - KHỞI CHẠY HỆ THỐNG")
    print("=" * 50)
    print("1. Đấu với AI MCTS (Bạn cầm X - đi trước)")
    print("2. Đấu với AI MCTS (Bạn cầm O - AI đi trước)")
    print("3. Chế độ 2 người chơi (PvP)")
    choice = input("Chọn chế độ (mặc định 1): ").strip()

    if choice == "2":
        run_game(mode="pve", simulations=800, human_side=O)
    elif choice == "3":
        run_game(mode="pvp")
    else:
        run_game(mode="pve", simulations=800, human_side=X)


if __name__ == "__main__":
    main()