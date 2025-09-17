import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# 1. 数据加载与预处理（已删除峰值位置）
# ---------------------------
def load_and_preprocess_data(filepath):
    # 从CSV文件加载数据
    df = pd.read_csv(filepath,encoding = 'GB2312')
    
    # 提取输入特征 (T, V, A, D, W, C)
    X = df[['T', 'V', 'A', 'D', 'W', 'C']].values  # 输入维度保持6列
    
    # 提取输出特征（已移除peak_location）
    y_ice_type = df['ice_type'].map({'流向冰': 0, '角状冰': 1}).values
    y_reg = df[['area', 'avg_thickness', 'max_thickness']].values  # 只保留前三个回归目标
    
    # 输入特征标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 回归目标标准化（全部使用Z-score）
    scaler_reg = StandardScaler()
    y_reg_scaled = scaler_reg.fit_transform(y_reg)
    
    return X_scaled, y_ice_type, y_reg_scaled, scaler_X, scaler_reg

# ---------------------------
# 2. 定义神经网络模型（输出层已修改）
# ---------------------------
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # 共享特征提取层
    x = Dense(32, activation='relu', kernel_regularizer='l2')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(16, activation='relu', kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 多任务输出分支（回归输出改为3个）
    ice_type_out = Dense(1, activation='sigmoid', name='ice_type')(x)
    reg_out = Dense(3, activation='linear', name='regression')(x)  # 修改为3个输出
    
    model = Model(inputs=inputs, outputs=[ice_type_out, reg_out])
    return model

# ---------------------------
# 3. 训练与评估（适配新输出维度）
# ---------------------------
def train_and_evaluate(X, y_ice_type, y_reg):
    # 划分训练集和验证集
    X_train, X_val, y_ice_train, y_ice_val, y_reg_train, y_reg_val = train_test_split(
        X, y_ice_type, y_reg, test_size=0.1, random_state=42
    )
    
    model = build_model(input_shape=(X.shape[1],))
    
    # 编译模型（回归目标维度变为3）
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'ice_type': 'binary_crossentropy', 'regression': 'mse'},
        metrics={'ice_type': 'accuracy', 'regression': 'mae'},
        loss_weights=[1.0, 1.0]
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    
    history = model.fit(
        X_train,
        {'ice_type': y_ice_train, 'regression': y_reg_train},
        epochs=200,
        batch_size=4,
        validation_data=(X_val, {'ice_type': y_ice_val, 'regression': y_reg_val}),
        callbacks=[early_stop],
        verbose=1
    )
    
    return model, history

# ---------------------------
# 4. 结果可视化（保持相同）
# ---------------------------
def plot_results(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['ice_type_loss'], label='Train Loss')
    plt.plot(history.history['val_ice_type_loss'], label='Val Loss')
    plt.title('Classification Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['regression_loss'], label='Train Loss')
    plt.plot(history.history['val_regression_loss'], label='Val Loss')
    plt.title('Regression Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# 新增可视化函数
# ---------------------------
def plot_predictions(model, X, y_ice_true, y_reg_scaled, scaler_reg):
    # 获取完整预测结果
    pred_ice_all, pred_reg_all = model.predict(X)
    
    # 反标准化回归数据
    y_reg_original = scaler_reg.inverse_transform(y_reg_scaled)
    pred_reg_original = scaler_reg.inverse_transform(pred_reg_all)
    
    plt.figure(figsize=(15, 10))
    
    # 回归预测对比子图
    regression_targets = ['Section Area (cm²)', 'Average Thickness (cm)', 'Maximum Thickness (cm)']
    for i in range(3):
        plt.subplot(2, 2, i+1)
        plt.scatter(y_reg_original[:, i], pred_reg_original[:, i], alpha=0.7, edgecolors='w')
        max_val = max(y_reg_original[:, i].max(), pred_reg_original[:, i].max())
        min_val = min(y_reg_original[:, i].min(), pred_reg_original[:, i].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title(f'{regression_targets[i]} Comparison')
        plt.grid(True, linestyle='--', alpha=0.5)
    
    # 分类概率分布子图
    plt.subplot(2, 2, 4)
    for label in [0, 1]:
        mask = (y_ice_true == label)
        plt.hist(pred_ice_all[mask].ravel(), 
                 bins=np.linspace(0, 1, 21), 
                 alpha=0.7, 
                 label=['Flow Ice', 'Angular Ice'][label])
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.title('Ice Type Classification Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    #plt.savefig('prediction_comparison.png', dpi=300)
    plt.show()
# ---------------------------
# 修改后的主程序
# ---------------------------
if __name__ == "__main__":
    data_path = "combine_1.csv"
    
    # 加载数据
    X, y_ice_type, y_reg, scaler_X, scaler_reg = load_and_preprocess_data(data_path)
    
    # 训练模型
    model, history = train_and_evaluate(X, y_ice_type, y_reg)
    plot_results(history)
    
    # 新增预测可视化
    plot_predictions(model, X, y_ice_type, y_reg, scaler_reg)  # 注意此处参数传递
    # 保存模型
    #model.save("ice_prediction_model.h5")
    print("模型已保存为 ice_prediction_model.h5")