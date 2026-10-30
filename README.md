# Core Environment – Caro 11×11

## 1. Giới thiệu

Module Core Environment chịu trách nhiệm xây dựng môi trường trò chơi Caro 11×11, quản lý bàn cờ, lượt chơi và các luật thắng/thua/hòa.

## 2. Quy ước bàn cờ

- Kích thước: 11×11.
- Ô trống: 0.
- Quân X: 1.
- Quân O: -1.
- Chỉ số hàng và cột bắt đầu từ 0.
- Người chơi X đi trước.
- Người thắng là người có ít nhất 5 quân liên tiếp theo hàng ngang, hàng dọc hoặc đường chéo.

## 3. Cách sử dụng

```python
from caro_environment import CaroEnv

env = CaroEnv()

# Tạo bàn cờ mới
board = env.reset()

# Thực hiện nước đi tại hàng 6, cột 6
result = env.step((5, 5))

# Hiển thị bàn cờ
env.render()
```

## 4. Các hàm chính

| Hàm | Chức năng |
|---|---|
| reset() | Khởi tạo bàn cờ |
| is_valid_move() | Kiểm tra nước đi |
| get_legal_moves() | Lấy các nước đi hợp lệ |
| step() | Thực hiện nước đi |
| check_winner() | Kiểm tra chiến thắng |
| is_draw() | Kiểm tra hòa |
| render() | Hiển thị bàn cờ |

## 5. Chạy kiểm thử

```bash
python test_environment.py
```
