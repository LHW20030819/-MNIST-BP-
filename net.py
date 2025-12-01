import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import cv2  # 用于处理自定义图像

# 启用 Keras 后端
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# --- 1. 数据集加载与预处理 ---


def load_and_preprocess_data():
    """加载 MNIST 数据集并进行预处理"""
    print("--- 1. 数据集加载与预处理 ---")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 归一化 (将像素值从 0-255 缩放到 0-1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 标签独热编码 (One-Hot Encoding)
    # 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # 注意：BP/MLP 需要展平(Flatten)输入，Keras 的 Flatten 层可以自动完成
    
    print(f"训练数据形状 (X): {x_train.shape}")
    print(f"训练标签形状 (Y): {y_train.shape}")
    
    return x_train, y_train, x_test, y_test

# --- 2. BP 神经网络模型构建 ---


def build_bp_model(input_shape):
    """构建 BP 神经网络 (MLP) 模型"""
    print("--- 2. BP 神经网络模型构建 ---")
    model = Sequential([
        # 展平层：将 28x28 图像输入转换为 784 维向量
        Flatten(input_shape=input_shape), 
        
        # 隐藏层 1: 512 个神经元，使用 ReLU 激活函数
        Dense(512, activation='relu'), 
        
        # 隐藏层 2: 256 个神经元，使用 ReLU 激活函数
        Dense(256, activation='relu'), 
        
        # 输出层: 10 个神经元 (对应 0-9 十个类别), 使用 Softmax 激活函数进行概率输出
        Dense(10, activation='softmax') 
    ])

    # 编译模型：
    # 优化器: Adam (高效的梯度下降变体)
    # 损失函数: Categorical Crossentropy (适用于 One-Hot 编码的分类问题)
    # 评估指标: 准确率 (Accuracy)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

# --- 3. 模型训练与评估 ---


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, 
                             epochs=10, batch_size=128):
    """训练模型并评估性能"""
    print("--- 3. 模型训练与评估 ---")
    
    # 训练模型
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        verbose=1)

    # 评估模型在测试集上的性能
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print("\n--- 评估结果 ---")
    print(f"测试集损失 (Loss): {loss:.4f}")
    print(f"测试集准确率 (Accuracy): {acc*100:.2f}%")
    
    return history

# --- 4. 识别你手写的数字 (核心要求) ---


def preprocess_custom_image(image_path):
    """
    预处理单个自定义手写数字图像，使其与 MNIST 格式兼容。
    1. 读入灰度图。
    2. 缩放至 28x28。
    3. 反色处理 (MNIST 笔画为白色，背景为黑色)。
    4. 归一化。
    """
    # 读入灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")

    # 确保背景是白色，笔画是黑色 (反色处理，使其与 MNIST 兼容)
    # 假设你的手写图是白底黑字，需要反转为黑底白字
    img = cv2.bitwise_not(img)
    
    # 缩放至 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # 归一化并转换为浮点数
    img = img.astype('float32') / 255.0

    # 确保背景足够黑 (接近 0)，前景足够亮 (接近 1)
    # 可以选择性地添加阈值处理 (可选)
    # _, img = cv2.threshold(img, 0.1, 1.0, cv2.THRESH_BINARY) 

    # Keras 需要 (Batch_Size, Height, Width) 的输入形状，这里是 (1, 28, 28)
    return np.expand_dims(img, axis=0) 


def test_custom_numbers(model):
    """
    测试你手写的 10 个数字 (0-9)。
    你需要创建 10 个图像文件，例如 '0.png', '1.png', ..., '9.png'，
    并确保它们位于当前目录下，且是黑白手写数字。
    """
    print("\n--- 4. 识别你手写的数字 ---")
    custom_results = {}
    
    # 定义 10 个自定义图像的文件名
    image_names = [f"{i}.png" for i in range(10)] 

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i, name in enumerate(image_names):
        try:
            # 预处理图像
            custom_img_processed = preprocess_custom_image(name)
            
            # 模型预测
            prediction = model.predict(custom_img_processed, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # 记录结果
            result_str = f"prediction: {predicted_class} ({confidence:.2f}%)"
            is_correct = predicted_class == i
            custom_results[i] = (result_str, is_correct)
            
            # 绘图展示
            ax = axes[i]
            # 移除批次维度，显示 28x28 图像
            ax.imshow(custom_img_processed[0], cmap='gray') 
            ax.set_title(f"Target: {i}\n{result_str}", 
                         color='green' if is_correct else 'red')
            ax.axis('off')

        except FileNotFoundError:
            print(f"❗ 警告: 缺少文件 {name}。请创建 10 个手写数字图像文件 (0.png - 9.png)。")
            custom_results[i] = ("文件缺失", False)
            axes[i].set_title(f"Target: {i}\n文件缺失", color='blue')
            axes[i].axis('off')
        except Exception as e:
            print(f"处理文件 {name} 时出错: {e}")
            custom_results[i] = (f"处理错误: {e}", False)
            axes[i].set_title(f"Target: {i}\n处理错误", color='red')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

    # 总结识别结果
    print("\n--- 识别结果总结 ---")
    all_correct = True
    for target, (result_str, is_correct) in custom_results.items():
        if not is_correct:
            all_correct = False
            print(f"❌ 目标 {target}: {result_str}")
        else:
            print(f"✅ 目标 {target}: {result_str}")
    
    if all_correct:
        print("\n🎉 恭喜！所有 10 个手写数字均被正确识别。")
    else:
        print("\n⚠️ 至少有一个数字识别错误。可能需要调整图像预处理或重新训练模型。")

# --- 主程序入口 ---


if __name__ == '__main__':
    # 1. 数据集加载
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # 2. 模型构建
    # 输入形状是 (28, 28)
    model = build_bp_model(x_train.shape[1:]) 
    
    # 3. 模型训练
    # 推荐使用 10-20 个 Epochs
    train_and_evaluate_model(model, x_train, y_train, x_test, y_test, 
                             epochs=10) 
    
    # 4. 测试自定义手写数字
    test_custom_numbers(model)