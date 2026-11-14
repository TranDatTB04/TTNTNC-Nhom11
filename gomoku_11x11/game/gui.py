import pygame
import sys
try:
    # Dùng khi chạy từ thư mục gốc (như play_vs_mcts.py hoặc main.py)
    from game.env import GomokuEnv
except ImportError:
    # Dùng khi chạy trực tiếp file gui.py từ bên trong thư mục game/
    from env import GomokuEnv

# Các hằng số đồ họa
BLUE = (0, 0, 255)
RED = (220, 0, 0)
BOARD_SIZE = 11
CELL_SIZE = 50      # Khoảng cách giữa các đường kẻ
MARGIN = 40         # Lề xung quanh bàn cờ
WINDOW_SIZE = CELL_SIZE * BOARD_SIZE + 2 * MARGIN

# Bảng màu
WOOD_COLOR = (222, 184, 135)
LINE_COLOR = (0, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def draw_board(screen):
    """Vẽ nền gỗ và lưới 11x11 ô vuông"""
    screen.fill(WOOD_COLOR)
    # Vẽ các đường kẻ (cần BOARD_SIZE + 1 đường để tạo ra BOARD_SIZE ô)
    for i in range(BOARD_SIZE + 1):
        # Đường ngang
        pygame.draw.line(screen, LINE_COLOR, 
                         (MARGIN, MARGIN + i * CELL_SIZE), 
                         (MARGIN + BOARD_SIZE * CELL_SIZE, MARGIN + i * CELL_SIZE), 2)
        # Đường dọc
        pygame.draw.line(screen, LINE_COLOR, 
                         (MARGIN + i * CELL_SIZE, MARGIN), 
                         (MARGIN + i * CELL_SIZE, MARGIN + BOARD_SIZE * CELL_SIZE), 2)

def draw_stones(screen, state):
    """Quét ma trận trạng thái và vẽ X / O vào giữa các ô vuông"""
    offset = CELL_SIZE // 2 - 8  
    thickness = 4                

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if state[row, col] != 0:
                # Dịch chuyển tọa độ vào chính giữa ô vuông
                center_x = MARGIN + col * CELL_SIZE + (CELL_SIZE // 2)
                center_y = MARGIN + row * CELL_SIZE + (CELL_SIZE // 2)
                
                if state[row, col] == 1:
                    pygame.draw.line(screen, BLUE, 
                                     (center_x - offset, center_y - offset), 
                                     (center_x + offset, center_y + offset), thickness)
                    pygame.draw.line(screen, BLUE, 
                                     (center_x + offset, center_y - offset), 
                                     (center_x - offset, center_y + offset), thickness)
                elif state[row, col] == -1:
                    pygame.draw.circle(screen, RED, (center_x, center_y), offset, thickness)

def run_game():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Gomoku 11x11 - AlphaZero Project")
    font = pygame.font.SysFont('tahoma', 48)

    env = GomokuEnv(board_size=BOARD_SIZE)
    state = env.reset()
    done = False

    # VÒNG LẶP TRÒ CHƠI (Bước 6)
    while True:
        draw_board(screen)
        draw_stones(screen, state)

        # Hiển thị thông báo khi có người thắng
        # Hiển thị thông báo khi có người thắng
        if done:
            winner_text = "Người chơi X thắng!" if env.turn == 1 else "Người chơi O thắng!"
            text_surf = font.render(winner_text, True, (255, 0, 0))
            screen.blit(text_surf, (WINDOW_SIZE//2 - text_surf.get_width()//2, WINDOW_SIZE//2 - 20))

        pygame.display.flip()

        # XỬ LÝ SỰ KIỆN CHUỘT
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not done:
                mouse_x, mouse_y = event.pos
                
                # Ánh xạ pixel thành tọa độ (row, col) bằng cách làm tròn đến giao điểm gần nhất
                col = round((mouse_x - MARGIN) // CELL_SIZE)
                row = round((mouse_y - MARGIN) // CELL_SIZE)

                # Kiểm tra xem click có nằm trong phạm vi bàn cờ không
                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                    action = row * BOARD_SIZE + col
                    
                    # Truyền hành động vào môi trường
                    new_state, reward, done = env.step(action)
                    
                    # Nếu trả về phần thưởng -10 nghĩa là đi đè lên quân khác, bỏ qua nước đi này
                    if reward == -10:
                        done = False  # Hủy bỏ cờ kết thúc do nước đi lỗi
                    else:
                        state = new_state

if __name__ == "__main__":
    run_game()