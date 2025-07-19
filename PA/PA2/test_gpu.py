import torch
import platform
def test_mac_gpu():
    if platform.system() != 'Darwin':
        print("当前系统不是 macOS")
        return False
    
    if not torch.backends.mps.is_available():
        print("当前 Mac 设备不支持 GPU 加速")
        return False
        
    try:
        device = torch.device("mps")
        test_tensor = torch.ones(1).to(device)
        print("GPU 测试成功！可以使用 MPS 设备进行加速")
        return True
    except Exception as e:
        print(f"GPU 测试失败，错误信息: {str(e)}")
        return False

if __name__ == "__main__":
    test_mac_gpu()
