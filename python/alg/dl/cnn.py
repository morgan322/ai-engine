import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# 卷积操作
def conv2d(inputs, filters, kernel_size, stride=1, padding=0):
    batch_size, in_channels, in_height, in_width = inputs.shape
    out_channels, _, kernel_height, kernel_width = filters.shape
    out_height = (in_height + 2 * padding - kernel_height) // stride + 1
    out_width = (in_width + 2 * padding - kernel_width) // stride + 1

    padded_inputs = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    outputs = np.zeros((batch_size, out_channels, out_height, out_width))

    for b in range(batch_size):
        for c_out in range(out_channels):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    h_start = h_out * stride
                    h_end = h_start + kernel_height
                    w_start = w_out * stride
                    w_end = w_start + kernel_width
                    patch = padded_inputs[b, :, h_start:h_end, w_start:w_end]
                    outputs[b, c_out, h_out, w_out] = np.sum(patch * filters[c_out])
    return outputs


# 最大池化操作
def max_pool2d(inputs, pool_size=2, stride=2):
    batch_size, in_channels, in_height, in_width = inputs.shape
    out_height = (in_height - pool_size) // stride + 1
    out_width = (in_width - pool_size) // stride + 1

    outputs = np.zeros((batch_size, in_channels, out_height, out_width))

    for b in range(batch_size):
        for c in range(in_channels):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    h_start = h_out * stride
                    h_end = h_start + pool_size
                    w_start = w_out * stride
                    w_end = w_start + pool_size
                    patch = inputs[b, c, h_start:h_end, w_start:w_end]
                    outputs[b, c, h_out, w_out] = np.max(patch)
    return outputs


# ReLU 激活函数
def relu(x):
    return np.maximum(0, x)


# 全连接层
def fully_connected(inputs, weights, biases):
    return np.dot(inputs, weights) + biases


# Softmax 函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# 交叉熵损失函数
def cross_entropy_loss(outputs, labels):
    num_samples = outputs.shape[0]
    return -np.sum(labels * np.log(outputs + 1e-8)) / num_samples


# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root='/home/morgan/ubt/data/ml', train=True, transform=transform, download=True)
test_dataset = MNIST(root='/home/morgan/ubt/data/ml', train=False, transform=transform, download=True)

train_images = []
train_labels = []
for image, label in train_dataset:
    train_images.append(image.numpy())
    train_labels.append(label)
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = []
test_labels = []
for image, label in test_dataset:
    test_images.append(image.numpy())
    test_labels.append(label)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# 数据预处理
train_images = train_images.reshape((-1, 1, 28, 28))
test_images = test_images.reshape((-1, 1, 28, 28))
train_labels = np.eye(10)[train_labels]

# 初始化参数
conv1_filters = np.random.randn(8, 1, 3, 3) * 0.01
fc1_weights = np.random.randn(8 * 13 * 13, 128) * 0.01
fc1_biases = np.zeros(128)
fc2_weights = np.random.randn(128, 10) * 0.01
fc2_biases = np.zeros(10)

# 训练参数
learning_rate = 0.01
num_epochs = 5
batch_size = 64
losses = []

# 训练循环
for epoch in range(num_epochs):
    num_batches = len(train_images) // batch_size
    for i in tqdm(range(num_batches), desc=f'Epoch {epoch + 1}/{num_epochs}'):
        batch_images = train_images[i * batch_size:(i + 1) * batch_size]
        batch_labels = train_labels[i * batch_size:(i + 1) * batch_size]

        # 前向传播
        conv1_out = conv2d(batch_images, conv1_filters, kernel_size=3)
        relu1_out = relu(conv1_out)
        pool1_out = max_pool2d(relu1_out)
        flattened = pool1_out.reshape((batch_size, -1))
        fc1_out = fully_connected(flattened, fc1_weights, fc1_biases)
        relu2_out = relu(fc1_out)
        fc2_out = fully_connected(relu2_out, fc2_weights, fc2_biases)
        output = softmax(fc2_out)

        # 计算损失
        loss = cross_entropy_loss(output, batch_labels)
        losses.append(loss)

        # 反向传播
        d_output = (output - batch_labels) / batch_size
        d_fc2_weights = np.dot(relu2_out.T, d_output)
        d_fc2_biases = np.sum(d_output, axis=0)
        d_relu2_out = np.dot(d_output, fc2_weights.T)
        d_fc1_out = d_relu2_out.copy()
        d_fc1_out[fc1_out <= 0] = 0
        d_fc1_weights = np.dot(flattened.T, d_fc1_out)
        d_fc1_biases = np.sum(d_fc1_out, axis=0)
        d_flattened = np.dot(d_fc1_out, fc1_weights.T)
        d_pool1_out = d_flattened.reshape(pool1_out.shape)

        # 池化层反向传播
        d_relu1_out = np.zeros_like(relu1_out)
        batch_size, in_channels, in_height, in_width = relu1_out.shape
        pool_size = 2
        stride = 2
        for b in range(batch_size):
            for c in range(in_channels):
                for h_out in range(in_height // pool_size):
                    for w_out in range(in_width // pool_size):
                        h_start = h_out * stride
                        h_end = h_start + pool_size
                        w_start = w_out * stride
                        w_end = w_start + pool_size
                        patch = relu1_out[b, c, h_start:h_end, w_start:w_end]
                        max_index = np.unravel_index(np.argmax(patch), patch.shape)
                        d_relu1_out[b, c, h_start + max_index[0], w_start + max_index[1]] = d_pool1_out[b, c, h_out, w_out]

        # 卷积层反向传播
        d_conv1_out = d_relu1_out.copy()
        d_conv1_out[conv1_out <= 0] = 0
        d_conv1_filters = np.zeros_like(conv1_filters)
        padded_inputs = np.pad(batch_images, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
        batch_size, in_channels, in_height, in_width = padded_inputs.shape
        out_channels, _, kernel_height, kernel_width = conv1_filters.shape
        out_height = (in_height - kernel_height) + 1
        out_width = (in_width - kernel_width) + 1
        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out
                        h_end = h_start + kernel_height
                        w_start = w_out
                        w_end = w_start + kernel_width
                        patch = padded_inputs[b, :, h_start:h_end, w_start:w_end]
                        # 检查索引是否越界
                        if h_out < d_conv1_out.shape[2] and w_out < d_conv1_out.shape[3]:
                            d_conv1_filters[c_out] += patch * d_conv1_out[b, c_out, h_out, w_out]

        # 参数更新
        conv1_filters -= learning_rate * d_conv1_filters
        fc1_weights -= learning_rate * d_fc1_weights
        fc1_biases -= learning_rate * d_fc1_biases
        fc2_weights -= learning_rate * d_fc2_weights
        fc2_biases -= learning_rate * d_fc2_biases

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss over Batches')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.show()

# 测试模型
correct_predictions = 0
predictions = []
for i in range(len(test_images)):
    image = test_images[i:i + 1]
    label = test_labels[i]

    # 前向传播
    conv1_out = conv2d(image, conv1_filters, kernel_size=3)
    relu1_out = relu(conv1_out)
    pool1_out = max_pool2d(relu1_out)
    flattened = pool1_out.reshape((1, -1))
    fc1_out = fully_connected(flattened, fc1_weights, fc1_biases)
    relu2_out = relu(fc1_out)
    fc2_out = fully_connected(relu2_out, fc2_weights, fc2_biases)
    output = softmax(fc2_out)

    predicted_label = np.argmax(output)
    predictions.append(predicted_label)
    if predicted_label == label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_images)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# 可视化部分测试结果
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Pred: {predictions[i]}, True: {test_labels[i]}')
    plt.axis('off')
plt.show()
    