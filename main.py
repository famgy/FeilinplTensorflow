import tensorflow as tf

from PIL import Image
from test_model import check_handwritten_digit

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(f'Hi, Tensorflow : ' + tf.__version__)

    # 加载数据集
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 构建机器学习模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # 配置和编译模型
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=5)

    # 保存模型为HDF5格式
    model.save("my_model.h5")

    # 在验证集上检查模型性能
    print("模型在验证集上的性能：")
    model.evaluate(x_test, y_test, verbose=2)

    # 对新样本进行预测
    new_samples = x_test[:5]
    predictions = model.predict(new_samples)
    print("模型对新样本的预测：")
    print(predictions)

    # Example usage
    image_path = "images/001.jpg"

    # Load the color image
    image_color = Image.open(image_path)

    # Convert to grayscale
    image_gray = image_color.convert("L")

    # Resize to 28x28
    image_gray_resized = image_gray.resize((28, 28))

    # Save the resized grayscale image
    image_gray_resized.save("images/resized_001.jpg")

    # 开始识别
    resized_image_path = "images/resized_001.jpg"
    predicted_class = check_handwritten_digit(resized_image_path)
    print("Predicted Digit:", predicted_class)
