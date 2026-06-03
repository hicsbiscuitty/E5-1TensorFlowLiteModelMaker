# E5-1TensorFlowLiteModelMaker
import tarfile
import numpy as np
import tensorflow as tf
from pathlib import Path

# ==========================================
# 1. 参数设置
# ==========================================
FLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
DATA_DIR = None                # 设为 None 则自动下载官方数据
EXPORT_DIR = "exported_flower_model"
EPOCHS = 5                     # 演示建议 5 轮，实际应用建议 15 轮以上
BATCH_SIZE = 32
IMAGE_SIZE = 224               # MobileNetV2 要求的标准尺寸
LEARNING_RATE = 1e-3
QUANTIZATION = "dynamic"       # 动态范围量化，大幅减小模型体积
SEED = 123

print("TensorFlow 版本:", tf.__version__)

# ==========================================
# 2. 读取并划分数据集 
# ==========================================
def load_flower_datasets(data_dir, image_size, batch_size, seed):
    if data_dir is None:
        # 下载并自动解压数据
        archive_path = tf.keras.utils.get_file("flower_photos.tgz", FLOWER_URL, extract=True)
        data_dir = Path(archive_path).parent / "flower_photos"
    else:
        data_dir = Path(data_dir)

    # 从文件夹结构自动生成带标签的数据集
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=seed,
        image_size=(image_size, image_size), batch_size=batch_size)
    
   
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=seed,
        image_size=(image_size, image_size), batch_size=batch_size)
    
    class_names = train_ds.class_names

    # 将验证集进一步拆分为 验证集 和 测试集
    val_batches = int(tf.data.experimental.cardinality(val_ds).numpy())
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)

    #使用缓存和预取提高训练速度
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    
    return train_ds, val_ds, test_ds, class_names

train_ds, val_ds, test_ds, class_names = load_flower_datasets(DATA_DIR, IMAGE_SIZE, BATCH_SIZE, SEED)
print("类别名称:", class_names)

# ==========================================
# 3. 构建 Keras 模型
# ==========================================
def build_model(num_classes, image_size, learning_rate):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")

    # MobileNetV2 将像素值归一化到 [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # 使用预训练的 MobileNetV2，去掉顶层分类器
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False, weights="imagenet", pooling="avg")

    # 冻结底座参数，仅训练添加的分类层
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)

    # 输出层：分类数量对应花卉种类
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])
    return model
    
model = build_model(len(class_names), IMAGE_SIZE, LEARNING_RATE)

# ==========================================
# 4. 训练与评估
# ==========================================
print("开始训练...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 最终测试集评估
loss, accuracy = model.evaluate(test_ds)
print(f"测试集准确率: {accuracy:.4f}")

# ==========================================
# 5. 模型转换与导出 
# ==========================================
def convert_to_tflite(model, quantization):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization == "dynamic":
        # 开启量化：大幅减小模型体积，加快移动端推理速度
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()

export_dir = Path(EXPORT_DIR)
export_dir.mkdir(parents=True, exist_ok=True)

# 保存标签文件 (部署时索引映射)
labels_path = export_dir / "labels.txt"
labels_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

# 保存 TFLite 模型
tflite_model = convert_to_tflite(model, QUANTIZATION)
tflite_path = export_dir / "model.tflite"
tflite_path.write_bytes(tflite_model)

print(f"TFLite 模型已导出至: {tflite_path}")

# ==========================================
# 6. 推理验证
# ==========================================
def smoke_test_tflite(tflite_path, test_ds, class_names):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 取 5 张图片做测试
    images, labels = next(iter(test_ds.unbatch().batch(5)))
    for i in range(5):
        img = np.expand_dims(images[i], axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], img)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details["index"])[0]
        predicted_id = np.argmax(prediction)
        print(f"真实类别: {class_names[labels[i]]} -> 预测类别: {class_names[predicted_id]}")

smoke_test_tflite(tflite_path, test_ds, class_names)
使用E4TFLClassify项目验证生成的model.tflite
<img width="1280" height="764" alt="联想截图_20260603145457" src="https://github.com/user-attachments/assets/fdba0152-d32c-4597-ba52-3a4234507a43" />
<img width="1080" height="2400" alt="Screenshot_20260603_145414" src="https://github.com/user-attachments/assets/7418d57a-47fe-420c-8897-b9ed1d9c78bd" />
