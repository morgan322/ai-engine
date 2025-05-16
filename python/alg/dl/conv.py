import numpy as np

def conv2d(input_data, kernel, stride=1, padding=0):
    """实现标准二维卷积"""
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, 0)
    if len(kernel.shape) == 2:
        kernel = np.expand_dims(np.expand_dims(kernel, 0), 0)
    
    C, H, W = input_data.shape
    out_channels, in_channels, k, _ = kernel.shape
    assert in_channels == C, f"输入通道数不匹配: {in_channels} != {C}"
    
    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2 * padding - k) // stride + 1
    
    padded = np.pad(input_data, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    output = np.zeros((out_channels, H_out, W_out))
    
    for oc in range(out_channels):
        for ic in range(in_channels):
            for i in range(H_out):
                for j in range(W_out):
                    window = padded[ic, i*stride:i*stride+k, j*stride:j*stride+k]
                    output[oc, i, j] += np.sum(window * kernel[oc, ic])
    
    return output if out_channels > 1 else output[0]

def conv2d_transpose(input_data, in_channels, out_channels, kernel_size, stride=1, padding=0):
    """实现转置卷积"""
    kernel = np.random.rand(out_channels, in_channels, kernel_size, kernel_size)
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, 0)
    
    C, H, W = input_data.shape
    out_channels_k, in_channels_k, k, _ = kernel.shape
    assert in_channels_k == C, f"输入通道数不匹配: {in_channels_k} != {C}"
    
    H_out = (H - 1) * stride - 2 * padding + k
    W_out = (W - 1) * stride - 2 * padding + k
    
    output = np.zeros((out_channels_k, H_out, W_out))
    
    for oc in range(out_channels_k):
        for ic in range(in_channels_k):
            dilated = np.zeros((H * stride, W * stride))
            dilated[::stride, ::stride] = input_data[ic]
            
            flipped_kernel = np.flip(np.flip(kernel[oc, ic], 0), 1)
            padded_dilated = np.pad(dilated, ((k-1, k-1), (k-1, k-1)), mode='constant')
            
            for i in range(H_out):
                for j in range(W_out):
                    window = padded_dilated[i:i+k, j:j+k]
                    output[oc, i, j] += np.sum(window * flipped_kernel)
    
    return output if out_channels_k > 1 else output[0]

def dilated_conv2d(input_data, kernel, dilation=1, padding=0):
    """实现空洞卷积（扩张卷积）"""
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, 0)
    if len(kernel.shape) == 2:
        kernel = np.expand_dims(np.expand_dims(kernel, 0), 0)
    
    C, H, W = input_data.shape
    out_channels, in_channels, k, _ = kernel.shape
    assert in_channels == C, f"输入通道数不匹配: {in_channels} != {C}"
    
    # 有效卷积核大小
    k_effective = k + (k - 1) * (dilation - 1)
    H_out = (H + 2 * padding - k_effective) + 1
    W_out = (W + 2 * padding - k_effective) + 1
    
    padded = np.pad(input_data, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    output = np.zeros((out_channels, H_out, W_out))
    
    for oc in range(out_channels):
        for ic in range(in_channels):
            for i in range(H_out):
                for j in range(W_out):
                    # 采样窗口，考虑dilation
                    window = np.zeros((k, k))
                    for ki in range(k):
                        for kj in range(k):
                            window[ki, kj] = padded[ic, i+ki*dilation, j+kj*dilation]
                    output[oc, i, j] += np.sum(window * kernel[oc, ic])
    
    return output if out_channels > 1 else output[0]

def depthwise_separable_conv2d(input_data, depthwise_kernel, pointwise_kernel, stride=1, padding=0):
    """实现深度可分离卷积"""
    # 深度卷积
    depthwise_output = conv2d(input_data, depthwise_kernel, stride, padding)
    # 逐点卷积
    pointwise_output = conv2d(depthwise_output, pointwise_kernel, 1, 0)
    return pointwise_output

def group_conv2d(input_data, kernel, groups=1, stride=1, padding=0):
    """实现分组卷积"""
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, 0)
    if len(kernel.shape) == 2:
        kernel = np.expand_dims(np.expand_dims(kernel, 0), 0)
    
    C, H, W = input_data.shape
    out_channels, in_channels, k, _ = kernel.shape
    assert in_channels % groups == 0, f"输入通道数必须能被组数整除: {in_channels} % {groups} != 0"
    assert out_channels % groups == 0, f"输出通道数必须能被组数整除: {out_channels} % {groups} != 0"
    
    group_in_channels = in_channels // groups
    group_out_channels = out_channels // groups
    
    output = np.zeros((out_channels, (H + 2 * padding - k) // stride + 1, (W + 2 * padding - k) // stride + 1))
    
    for g in range(groups):
        group_input = input_data[g*group_in_channels:(g+1)*group_in_channels]
        group_kernel = kernel[g*group_out_channels:(g+1)*group_out_channels, g*group_in_channels:(g+1)*group_in_channels]
        group_output = conv2d(group_input, group_kernel, stride, padding)
        output[g*group_out_channels:(g+1)*group_out_channels] = group_output
    
    return output

# 示例用法
if __name__ == "__main__":
    # 创建测试输入
    x = np.random.rand(3, 32, 32)  # 3通道，32x32图像
    
    # 标准卷积
    kernel = np.random.rand(16, 3, 3, 3)  # 16个3x3卷积核
    conv_output = conv2d(x, kernel, stride=1, padding=1)
    print(f"标准卷积输出形状: {conv_output.shape}")
    
    # 转置卷积
    deconv_output = conv2d_transpose(conv_output, 16, 3, 3, stride=2, padding=1)
    print(f"转置卷积输出形状: {deconv_output.shape}")
    
    # 空洞卷积
    dilated_kernel = np.random.rand(16, 3, 3, 3)
    dilated_output = dilated_conv2d(x, dilated_kernel, dilation=2, padding=2)
    print(f"空洞卷积输出形状: {dilated_output.shape}")
    
    # 深度可分离卷积
    depthwise_kernel = np.random.rand(3, 3, 3, 3)  # 每个通道一个卷积核
    pointwise_kernel = np.random.rand(16, 3, 1, 1)  # 1x1卷积核
    depthwise_output = depthwise_separable_conv2d(x, depthwise_kernel, pointwise_kernel, stride=1, padding=1)
    print(f"深度可分离卷积输出形状: {depthwise_output.shape}")
    
    # 分组卷积
    group_kernel = np.random.rand(18, 3, 3, 3)
    group_output = group_conv2d(x, group_kernel, groups=3, stride=1, padding=1)
    print(f"分组卷积输出形状: {group_output.shape}")    