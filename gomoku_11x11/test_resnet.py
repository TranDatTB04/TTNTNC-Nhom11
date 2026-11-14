
import torch
from environment.caro_environment import CaroEnv
from models.resnet import ResNet, encode_state


def test_resnet():

    # 1. Khởi tạo môi trường Caro
    env = CaroEnv()

    # 2. Khởi tạo ResNet
    model = ResNet(
        board_size=11,
        input_channels=2,
        num_res_blocks=5,
        num_channels=128
    )

    # 3. Lấy trạng thái ban đầu
    state = env.get_observation()
    current_player = env.current_player

    # 4. Kiểm tra mã hóa trạng thái
    encoded_state = encode_state(
        state,
        current_player
    )

    print("Encoded state:", encoded_state.shape)

    # 5. Kiểm tra Forward
    policy_logits, value = model(encoded_state)

    print("Policy shape:", policy_logits.shape)
    print("Value shape:", value.shape)

    # 6. Kiểm tra Predict với legal_moves từ CaroEnv
    legal_moves = env.get_legal_moves()

    policy_probs, state_value = model.predict(
        state,
        current_player,
        legal_moves
    )

    print("Policy probabilities:", policy_probs.shape)
    print("Value:", state_value)

    # 7. Kiểm tra nước đi có xác suất cao nhất
    best_action = policy_probs.argmax()

    row = best_action // env.board_size
    col = best_action % env.board_size

    print("Best action:", (row, col))
    print("Best probability:", policy_probs[best_action])

    # 8. Kiểm tra nước đi hợp lệ
    assert (row, col) in legal_moves

    print("\nRESNET TEST PASSED!")


if __name__ == "__main__":
    test_resnet()
