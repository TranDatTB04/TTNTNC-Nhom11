
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_state(state, current_player, board_size=11):
    """
    Chuyển trạng thái bàn cờ thành tensor đầu vào cho ResNet.

    Đầu vào:
        state: Ma trận bàn cờ (board_size, board_size).
               1  : Quân X
              -1  : Quân O
               0  : Ô trống

        current_player: Người đang đến lượt (1 hoặc -1).

    Đầu ra:
        Tensor có kích thước (2, board_size, board_size).

        Channel 0: Vị trí quân của người đang đến lượt.
        Channel 1: Vị trí quân của đối thủ.
    """
    board = np.asarray(state)

    if board.shape != (board_size, board_size):
        raise ValueError(
            f"Kích thước bàn cờ phải là "
            f"({board_size}, {board_size}), nhận được {board.shape}"
        )

    if current_player not in (1, -1):
        raise ValueError("current_player phải là 1 hoặc -1")

    if not np.isin(board, [-1, 0, 1]).all():
        raise ValueError("Trạng thái bàn cờ chỉ được chứa -1, 0 và 1")

    # Kênh 1: Quân của người đang đến lượt.
    current_plane = (board == current_player).astype(np.float32)

    # Kênh 2: Quân của đối thủ.
    opponent_plane = (board == -current_player).astype(np.float32)

    # Ghép thành tensor (2, 11, 11).
    encoded_state = np.stack(
        [current_plane, opponent_plane],
        axis=0
    )

    return torch.from_numpy(encoded_state)


class ResidualBlock(nn.Module):
    """
    Một khối Residual gồm:
    Conv2D -> BatchNorm -> ReLU
    Conv2D -> BatchNorm
    Cộng với đầu vào ban đầu (Skip Connection).
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip Connection.
        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet dành cho Gomoku 11x11.

    Đầu vào:
        (batch_size, 2, 11, 11)

    Đầu ra:
        Policy: (batch_size, 121) - Policy logits.
        Value : (batch_size,)     - Giá trị trạng thái [-1, 1].
    """

    def __init__(
        self,
        board_size=11,
        input_channels=2,
        num_res_blocks=5,
        num_channels=128
    ):
        super(ResNet, self).__init__()

        self.board_size = board_size
        self.action_size = board_size * board_size

        # ==========================================
        # 1. CONVOLUTIONAL STEM
        # ==========================================

        self.initial_conv = nn.Conv2d(
            input_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.initial_bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

        # ==========================================
        # 2. RESIDUAL BLOCKS
        # ==========================================

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(num_channels)
                for _ in range(num_res_blocks)
            ]
        )

        # ==========================================
        # 3. POLICY HEAD
        # ==========================================
        # Dự đoán điểm số cho 121 vị trí trên bàn cờ.

        self.policy_conv = nn.Conv2d(
            num_channels,
            2,
            kernel_size=1,
            bias=False
        )

        self.policy_bn = nn.BatchNorm2d(2)

        self.policy_fc = nn.Linear(
            2 * board_size * board_size,
            self.action_size
        )

        # ==========================================
        # 4. VALUE HEAD
        # ==========================================
        # Đánh giá trạng thái bàn cờ trong khoảng [-1, 1].

        self.value_conv = nn.Conv2d(
            num_channels,
            1,
            kernel_size=1,
            bias=False
        )

        self.value_bn = nn.BatchNorm2d(1)

        self.value_fc1 = nn.Linear(
            board_size * board_size,
            256
        )

        self.value_fc2 = nn.Linear(256, 1)

        # ==========================================
        # 5. KHỞI TẠO TRỌNG SỐ
        # ==========================================

        self._initialize_weights()

    def _initialize_weights(self):
        """Khởi tạo trọng số cho các lớp mạng."""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Lan truyền dữ liệu qua mạng ResNet.

        Trả về:
            policy_logits: Điểm số của 121 nước đi.
            value: Giá trị đánh giá trạng thái.
        """

        # Hỗ trợ cả đầu vào (2, 11, 11)
        # và đầu vào theo batch (B, 2, 11, 11).
        device = next(self.parameters()).device

        x = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=device
        )

        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.dim() != 4:
            raise ValueError(
                "Đầu vào phải có dạng (2, H, W) "
                "hoặc (B, 2, H, W)"
            )

        expected_shape = (
            2,
            self.board_size,
            self.board_size
        )

        if tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                f"Đầu vào phải có dạng (B, {expected_shape}), "
                f"nhận được {tuple(x.shape)}"
            )

        # ==========================================
        # CONVOLUTIONAL STEM
        # ==========================================

        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = self.relu(out)

        # ==========================================
        # RESIDUAL NETWORK
        # ==========================================

        out = self.residual_blocks(out)

        # ==========================================
        # POLICY HEAD
        # ==========================================

        policy = self.policy_conv(out)
        policy = self.policy_bn(policy)
        policy = self.relu(policy)

        policy = policy.flatten(start_dim=1)
        policy_logits = self.policy_fc(policy)

        # ==========================================
        # VALUE HEAD
        # ==========================================

        value = self.value_conv(out)
        value = self.value_bn(value)
        value = self.relu(value)

        value = value.flatten(start_dim=1)
        value = self.value_fc1(value)
        value = self.relu(value)
        value = self.value_fc2(value)

        # Giá trị nằm trong khoảng [-1, 1].
        value = torch.tanh(value).squeeze(-1)

        return policy_logits, value

    @torch.no_grad()
    def predict(self, state, current_player, legal_actions=None):
        """
        Dự đoán Policy và Value cho một trạng thái bàn cờ.

        Tham số:
            state: Ma trận bàn cờ.
            current_player: Người đang đến lượt.
            legal_actions: Danh sách action hợp lệ dạng số nguyên.
                           Có thể lấy từ env.get_valid_moves().

        Trả về:
            policy_probs: Mảng xác suất cho 121 hành động.
            value: Giá trị trạng thái trong [-1, 1].
        """

        # Mã hóa trạng thái.
        encoded_state = encode_state(
            state,
            current_player,
            self.board_size
        )

        device = next(self.parameters()).device
        encoded_state = encoded_state.unsqueeze(0).to(device)

        # Chuyển sang chế độ đánh giá.
        was_training = self.training
        self.eval()

        try:
            policy_logits, value = self.forward(encoded_state)

            # Chỉ phân phối xác suất trên các nước đi hợp lệ.
            if legal_actions is not None:
                legal_actions = list(legal_actions)

                if not legal_actions:
                    raise ValueError(
                        "Không còn nước đi hợp lệ trên bàn cờ"
                    )

                indices = torch.as_tensor(
                    legal_actions,
                    dtype=torch.long,
                    device=device
                )

                if torch.any(indices < 0) or torch.any(
                    indices >= self.action_size
                ):
                    raise ValueError("legal_actions chứa action không hợp lệ")

                masked_logits = torch.full_like(
                    policy_logits,
                    float("-inf")
                )

                masked_logits[0, indices] = policy_logits[0, indices]
                policy_logits = masked_logits

            # Chuyển Policy logits thành xác suất.
            policy_probs = F.softmax(
                policy_logits,
                dim=1
            )[0]

            return (
                policy_probs.cpu().numpy(),
                float(value.item())
            )

        finally:
            # Khôi phục trạng thái train/eval ban đầu.
            self.train(was_training)
