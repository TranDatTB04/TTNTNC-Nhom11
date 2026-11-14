import pygame
import sys
from game.env import GomokuEnv
from game.gui import draw_board, draw_stones, BOARD_SIZE, CELL_SIZE, MARGIN, WINDOW_SIZE
from mcts.mcts import MCTS

def play_human_vs_ai():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Cờ Caro 11x11: Người (X) vs MCTS AI (O)")
    font = pygame.font.SysFont('tahoma', 42)
    status_font = pygame.font.SysFont('tahoma', 24)

    # Khởi tạo môi trường và AI
    env = GomokuEnv(board_size=BOARD_SIZE)
    state = env.reset()
    
    # Thiết lập AI MCTS với 1000 lượt mô phỏng cho mỗi nước đi
    # (Bạn có thể tăng lên 2000-3000 nếu muốn AI đánh khôn hơn, hoặc giảm xuống 500 nếu thấy máy nghĩ hơi lâu)
    ai_agent = MCTS(env, simulations=1000)

    HUMAN_PLAYER = 1   # Người chơi cầm quân X (đi trước)
    AI_PLAYER = -1     # AI cầm quân O (đi sau)
    done = False

    while True:
        draw_board(screen)
        draw_stones(screen, state)

        # Hiển thị trạng thái lượt đi ở góc trên màn hình
        if not done:
            if env.turn == HUMAN_PLAYER:
                status_text = "Lượt của bạn (X) - Hãy click vào ô trống"
                status_color = (0, 100, 0)
            else:
                status_text = "AI (O) đang tính toán nước đi..."
                status_color = (200, 0, 0)
            status_surf = status_font.render(status_text, True, status_color)
            screen.blit(status_surf, (MARGIN, 10))
        else:
            # Thông báo kết quả khi ván đấu kết thúc
            if len(env.get_valid_moves()) == 0 and not env._check_win(0, 0): # Check tạm hòa
                winner_text = "Hòa cờ!"
            else:
                winner_text = "Bạn (X) đã thắng!" if env.turn == HUMAN_PLAYER else "AI (O) đã thắng!"
            
            text_surf = font.render(winner_text, True, (255, 0, 0))
            # Vẽ nền trắng mờ phía sau chữ thông báo cho dễ đọc
            bg_rect = text_surf.get_rect(center=(WINDOW_SIZE//2, WINDOW_SIZE//2))
            pygame.draw.rect(screen, (255, 255, 255), bg_rect.inflate(20, 10))
            screen.blit(text_surf, bg_rect)

        pygame.display.flip()

        # LƯỢT CỦA AI (Tự động chạy khi đến lượt -1 và game chưa kết thúc)
        if not done and env.turn == AI_PLAYER:
            # Gọi bộ tìm kiếm MCTS để chọn nước đi
            ai_action = ai_agent.search(state, AI_PLAYER)
            state, reward, done = env.step(ai_action)
            continue # Quay lại đầu vòng lặp để vẽ ngay nước đi của AI

        # XỬ LÝ SỰ KIỆN CỦA NGƯỜI CHƠI
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Chỉ nhận click chuột khi đang là lượt của Người chơi và ván chưa kết thúc
            if event.type == pygame.MOUSEBUTTONDOWN and not done and env.turn == HUMAN_PLAYER:
                mouse_x, mouse_y = event.pos
                
                col = (mouse_x - MARGIN) // CELL_SIZE
                row = (mouse_y - MARGIN) // CELL_SIZE

                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                    action = row * BOARD_SIZE + col
                    new_state, reward, step_done = env.step(action)
                    
                    if reward != -10: # Nếu nước đi hợp lệ
                        state = new_state
                        done = step_done

if __name__ == "__main__":
    play_human_vs_ai()