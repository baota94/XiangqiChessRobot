import torch
import subprocess
import sys

def check_cuda():
    print("=" * 60)
    print("🔍 KIỂM TRA PHIÊN BẢN CUDA VÀ GPU TRÊN HỆ THỐNG")
    print("=" * 60)

    # Kiểm tra có GPU CUDA không
    cuda_available = torch.cuda.is_available()
    print(f"\n✅ CUDA khả dụng: {cuda_available}")

    if not cuda_available:
        print("⚠️  Không phát hiện GPU CUDA.")
        print("👉 Kiểm tra lại driver NVIDIA hoặc cài đặt CUDA Toolkit.")
        print("   Gợi ý cài PyTorch hỗ trợ GPU (CUDA 12.1):")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n")
        return

    # Lấy thông tin GPU
    device_count = torch.cuda.device_count()
    print(f"🧠 Số GPU phát hiện: {device_count}")

    for i in range(device_count):
        print(f"  ├─ GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  │  - Tổng bộ nhớ: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

    # Phiên bản CUDA runtime và driver
    print("\n📦 Phiên bản PyTorch:")
    print(f"  torch: {torch.__version__}")

    cuda_ver = torch.version.cuda
    print(f"  CUDA (PyTorch build): {cuda_ver}")

    try:
        nvcc_out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
        print("\n🧰 Thông tin CUDA Toolkit (nvcc):")
        print(nvcc_out.strip().split("\n")[-1])
    except FileNotFoundError:
        print("\n⚠️  Lệnh 'nvcc' không tồn tại — có thể bạn chưa cài CUDA Toolkit đầy đủ.")
        print("   Tải tại: https://developer.nvidia.com/cuda-downloads")

    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
        first_line = nvidia_smi.split("\n")[2]
        print("\n💻 NVIDIA Driver Info:")
        print(first_line)
    except Exception as e:
        print(f"\n⚠️  Không chạy được nvidia-smi: {e}")

    # Gợi ý nâng cấp nếu CUDA cũ
    if cuda_ver:
        major = int(cuda_ver.split(".")[0])
        if major < 12:
            print("\n⚙️  Gợi ý cập nhật: CUDA bạn đang dùng hơi cũ.")
            print("   Gợi ý nâng cấp PyTorch + CUDA 12.1:")
            print("   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    print("\n✅ Kiểm tra hoàn tất.\n")


if __name__ == "__main__":
    check_cuda()
