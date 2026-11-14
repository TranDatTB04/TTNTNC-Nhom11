import pygame
import sys

try:
    # Dùng khi chạy từ thư mục gốc (ví dụ: python main.py)
    from environment.caro_environment import CaroEnv, BOARD_SIZE, EMPTY, X, O
except ImportError:
    # Dùng khi chạy trực tiếp từ bên trong thư mục environment/
    from caro_environment import CaroEnv, BOARD_SIZE, EMPTY, X, O

# Các hằng số đồ họa (Đánh nằm gọn trong ô vuông 11x11)
CELL_SIZE = 50
MARGIN = 40
WINDOW_SIZE = CELL_SIZE * BOARD_SIZE + 2 * MARGIN

# Bảng màu
WOOD_COLOR = (222, 184, 135)
LINE_COLOR = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (220, 0, 0)
HIGHLIGHT_COLOR = (255, 255, 0)


def draw_board(screen):
    """Vẽ nền bàn cờ và lưới 11x11 ô vuông."""
    screen.fill(WOOD_COLOR)
    for i in range(BOARD_SIZE + 1):
        # Đường ngang
        pygame.draw.line(
            screen,
            LINE_COLOR,
            (MARGIN, MARGIN + i * CELL_SIZE),
            (MARGIN + BOARD_SIZE * CELL_SIZE, MARGIN + i * CELL_SIZE),
            2,
        )
        # Đường dọc
        pygame.draw.line(
            screen,
            LINE_COLOR,
            (MARGIN + i * CELL_SIZE, MARGIN),
            (MARGIN + i * CELL_SIZE, MARGIN + BOARD_SIZE * CELL_SIZE),
            2,
        )


def draw_stones(screen, board, last_move=None):
    """Quét ma trận board (list lồng nhau) và vẽ X / O vào giữa các ô vuông."""
    offset = CELL_SIZE // 2 - 8
    thickness = 4

    # Tô sáng nhẹ ô vừa đánh cuối cùng để dễ quan sát
    if last_move is not None:
        lr, lc = last_move
        highlight_rect = pygame.Rect(
            MARGIN + lc * CELL_SIZE + 2,
            MARGIN + lr * CELL_SIZE + 2,
            CELL_SIZE - 3,
            CELL_SIZE - 3,
        )
        pygame.draw.rect(screen, (238, 208, 165), highlight_rect)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cell = board[row][col]
            if cell != EMPTY:
                center_x = MARGIN + col * CELL_SIZE + (CELL_SIZE // 2)
                center_y = MARGIN + row * CELL_SIZE + (CELL_SIZE // 2)

                if cell == X:
                    pygame.draw.line(
                        screen,
                        BLUE,
                        (center_x - offset, center_y - offset),
                        (center_x + offset, center_y + offset),
                        thickness,
                    )
                    pygame.draw.line(
                        screen,
                        BLUE,
                        (center_x + offset, center_y - offset),
                        (center_x - offset, center_y + offset),
                        thickness,
                    )
                elif cell == O:
                    pygame.draw.circle(screen, RED, (center_x, center_y), offset, thickness)


def run_game():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Cờ Caro 11x11 - Chế độ 2 Người chơi")
    
    # Sử dụng font hệ thống hỗ trợ đầy đủ Unicode tiếng Việt
    font = pygame.font.SysFont("Times New Roman", 36, bold=True)
    status_font = pygame.font.SysFont("Times New Roman", 22)

    env = CaroEnv()
    board = env.reset()

    while True:
        draw_board(screen)
        draw_stones(screen, board, env.last_move)

        # Thanh trạng thái phía trên bàn cờ
        if not env.done:
            turn_text = "Lượt hiện tại: Người chơi X (Xanh)" if env.current_player == X else "Lượt hiện tại: Người chơi O (Đỏ)"
            turn_color = BLUE if env.current_player == X else RED
            status_surf = status_font.render(turn_text, True, turn_color)
            screen.blit(status_surf, (MARGIN, 10))
        else:
            # Thông báo kết quả khi kết thúc ván
            if env.winner == X:
                end_text = "Người chơi X thắng! (Nhấn R để chơi lại)"
            elif env.winner == O:
                end_text = "Người chơi O thắng! (Nhấn R để chơi lại)"
            else:
                end_text = "Hòa cờ! (Nhấn R để chơi lại)"

            text_surf = font.render(end_text, True, (200, 0, 0))
            bg_rect = text_surf.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
            pygame.draw.rect(screen, (255, 255, 255), bg_rect.inflate(24, 14))
            pygame.draw.rect(screen, (0, 0, 0), bg_rect.inflate(24, 14), 2)
            screen.blit(text_surf, bg_rect)

        pygame.display.flip()

        # Xử lý sự kiện bàn phím và chuột
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # 1. Reset bằng phím R (bắt cả mã phím lẫn ký tự không phân biệt hoa/thường và Unikey) hoặc phím Space
            if event.type == pygame.KEYDOWN:
                if (
                    event.key in (pygame.K_r, pygame.K_SPACE)
                    or event.unicode.lower() == "r"
                ):
                    board = env.reset()

            # 2. Xử lý click chuột
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Nếu ván đã kết thúc, click chuột bất kỳ đâu trên màn hình cũng sẽ bắt đầu ván mới
                if env.done:
                    board = env.reset()
                else:
                    mouse_x, mouse_y = event.pos
                    col = (mouse_x - MARGIN) // CELL_SIZE
                    row = (mouse_y - MARGIN) // CELL_SIZE

                    next_board, reward, terminated, truncated, info = env.step((row, col))
                    if reward != -1:
                        board = next_board


if __name__ == "__main__":
    run_game()