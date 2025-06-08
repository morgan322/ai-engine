import numpy as np

def conv2d(inputs, filters, stride=1, padding=0, dilation=1):
    """
    实现标准2D卷积操作: 单个输出通道的结果通过对所有输入通道的卷积结果求和得到；
    """
    batch_size, in_channels, in_height, in_width = inputs.shape
    out_channels, _, kernel_h, kernel_w = filters.shape
    
    # 计算输出尺寸
    out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    
    # 填充输入
    padded = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    # 初始化输出
    output = np.zeros((batch_size, out_channels, out_height, out_width))
    
    # 执行卷积
    for b in range(batch_size):
        for oc in range(out_channels):
            for ic in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # 计算卷积窗口
                        window = padded[b, ic, 
                                      i*stride:i*stride+dilation*kernel_h:dilation, 
                                      j*stride:j*stride+dilation*kernel_w:dilation]
                        # 执行卷积操作
                        output[b, oc, i, j] += np.sum(window * filters[oc, ic])
    return output

def conv2d_transpose(inputs, filters, stride=1, padding=0, output_padding=0):
    """
    实现2D转置卷积操作
    """
    batch_size, in_channels, in_height, in_width = inputs.shape
    out_channels, _, kernel_h, kernel_w = filters.shape
    
    # 计算输出尺寸
    out_height = (in_height - 1) * stride - 2 * padding + kernel_h + output_padding
    out_width = (in_width - 1) * stride - 2 * padding + kernel_w + output_padding
    
    # 初始化输出
    output = np.zeros((batch_size, out_channels, out_height, out_width))
    
    # 对每个样本执行转置卷积
    for b in range(batch_size):
        for oc in range(out_channels):
            for ic in range(in_channels):
                # 对输入的每个位置
                for i in range(in_height):
                    for j in range(in_width):
                        # 计算在输出中的位置
                        out_i_start = i * stride
                        out_j_start = j * stride
                        
                        # 获取卷积核并翻转
                        flipped_kernel = np.flip(filters[oc, ic], axis=(0, 1))
                        
                        # 将卷积核的值添加到输出的相应位置
                        for ki in range(kernel_h):
                            for kj in range(kernel_w):
                                out_i = out_i_start + ki
                                out_j = out_j_start + kj
                                
                                # 检查是否在有效范围内
                                if 0 <= out_i < out_height and 0 <= out_j < out_width:
                                    output[b, oc, out_i, out_j] += inputs[b, ic, i, j] * flipped_kernel[ki, kj]
    return output


def depthwise_separable_conv2d(input_tensor, depthwise_filter, pointwise_filter, stride=1, padding=0):
    """
    实现深度可分离二维卷积
    
    参数:
    input_tensor: 输入张量，形状为 [batch_size, in_channels, height, width]
    depthwise_filter: 深度卷积滤波器，形状为 [in_channels, filter_height, filter_width]
    pointwise_filter: 逐点卷积滤波器，形状为 [out_channels, in_channels]
    stride: 步长，默认为1
    padding: 填充大小，默认为0
    
    返回:
    输出张量，形状为 [batch_size, out_channels, out_height, out_width]
    """
    # 获取输入尺寸
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    _, filter_height, filter_width = depthwise_filter.shape
    out_channels, _ = pointwise_filter.shape
    
    # 计算输出尺寸
    out_height = (in_height + 2 * padding - filter_height) // stride + 1
    out_width = (in_width + 2 * padding - filter_width) // stride + 1
    
    # 填充输入（为每个样本添加填充）
    padded_input = np.pad(
        input_tensor, 
        ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
        mode='constant'
    )
    
    # 初始化深度卷积输出张量 [batch_size, in_channels, out_height, out_width]
    depthwise_output = np.zeros((batch_size, in_channels, out_height, out_width))
    
    # 深度卷积: 每个通道使用独立的滤波器
    for b in range(batch_size):
        for c in range(in_channels):
            # 获取当前批次和通道的输入
            channel_input = padded_input[b, c]
            # 获取当前通道的滤波器
            channel_filter = depthwise_filter[c]
            
            for i in range(out_height):
                for j in range(out_width):
                    # 提取当前窗口
                    window = channel_input[
                        i*stride:i*stride+filter_height, 
                        j*stride:j*stride+filter_width
                    ]
                    # 应用滤波器并存储结果
                    depthwise_output[b, c, i, j] = np.sum(window * channel_filter)
    
    # 逐点卷积: 1x1卷积，合并通道
    output = np.zeros((batch_size, out_channels, out_height, out_width))
    
    for b in range(batch_size):
        for oc in range(out_channels):
            for ic in range(in_channels):
                # 逐点卷积：对每个输出通道，将所有输入通道加权求和
                output[b, oc] += depthwise_output[b, ic] * pointwise_filter[oc, ic]
    
    return output

def grouped_conv2d(inputs, filters, kernel_size, stride=1, padding=0, groups=1):
    """
    实现分组卷积操作
    """
    batch_size, in_channels, in_height, in_width = inputs.shape
    out_channels, _, kernel_h, kernel_w = filters.shape
    
    # 检查输入通道数和输出通道数是否能被组数整除
    assert in_channels % groups == 0, "输入通道数必须能被组数整除"
    assert out_channels % groups == 0, "输出通道数必须能被组数整除"
    
    # 每组的输入和输出通道数
    channels_per_group_in = in_channels // groups
    channels_per_group_out = out_channels // groups
    
    # 计算输出尺寸
    out_height = (in_height + 2 * padding - (kernel_h - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - (kernel_w - 1) - 1) // stride + 1
    
    # 填充输入
    padded = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    # 初始化输出
    output = np.zeros((batch_size, out_channels, out_height, out_width))
    
    # 执行分组卷积
    for g in range(groups):
        # 计算当前组的输入和输出通道范围
        in_start = g * channels_per_group_in
        in_end = (g + 1) * channels_per_group_in
        out_start = g * channels_per_group_out
        out_end = (g + 1) * channels_per_group_out
        
        # 获取当前组的输入和滤波器
        group_input = padded[:, in_start:in_end]
        group_filters = filters[out_start:out_end, in_start:in_end]
        
        # 对当前组执行卷积
        group_output = conv2d(group_input, group_filters, stride, 0)
        
        # 将结果放入总输出中
        output[:, out_start:out_end] += group_output
    
    return output

# 示例使用
if __name__ == "__main__":
    # 测试用输入
    batch_size = 1
    in_channels = 3
    height = 32
    width = 32
    out_channels = 3
    kernel_size = 3
    
    inputs = np.random.rand(batch_size, in_channels, height, width)
    print(f"输入形状: {inputs.shape}")
    
    # 测试标准卷积
    filters = np.random.rand(out_channels, in_channels, kernel_size, kernel_size)
    conv_output = conv2d(inputs, filters, stride=1, padding=1)
    print(f"标准卷积输出形状: {conv_output.shape}")
    
    # 测试转置卷积
    transpose_filters = np.random.rand(in_channels, out_channels, kernel_size, kernel_size)
    transpose_output = conv2d_transpose(conv_output, transpose_filters, stride=2, padding=1)
    print(f"转置卷积输出形状: {transpose_output.shape}")
    
    # 测试空洞卷积
    dilation = 2
    dilated_filters = np.random.rand(out_channels, in_channels, kernel_size, kernel_size)
    dilated_output = conv2d(inputs, dilated_filters, stride=1, padding=dilation, dilation=dilation)
    print(f"空洞卷积输出形状: {dilated_output.shape}")
    
    # 测试深度可分离卷积
    dw_filters = np.random.rand(in_channels, kernel_size, kernel_size)  # 每个输入通道一个滤波器
    pw_filters = np.random.rand(in_channels, out_channels)  # 逐点卷积滤波器
    ds_output = depthwise_separable_conv2d(inputs, dw_filters, pw_filters, stride=1, padding=1)
    print(f"深度可分离卷积输出形状: {ds_output.shape}")
    
    # 测试分组卷积
    groups = 3
    grouped_filters = np.random.rand(out_channels, in_channels, kernel_size, kernel_size)
    grouped_output = grouped_conv2d(inputs, grouped_filters, kernel_size, stride=1, padding=1, groups=groups)
    print(f"分组卷积输出形状: {grouped_output.shape}")    