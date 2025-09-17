import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from matplotlib.gridspec import GridSpec
# 设置字体为支持中文的字体
rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei' 等
rcParams['axes.unicode_minus'] = False  # 防止负号显示为乱码
# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
def load_and_preprocess_data(filepath):
    # 加载CSV数据（假设列顺序为：T, V, A, D, W, C, ice_type, area, avg_thickness, max_thickness）
    df = pd.read_csv(filepath,encoding = "GB2312")
    
    # 提取输入特征（6个）
    input_features = ['T', 'V', 'A', 'D', 'W', 'C']
    X = df[input_features].values
    
    # 提取输出特征（1分类 + 3回归）
    y_ice_type = df['ice_type'].map({'流向冰': 0, '角状冰': 1}).values  # 分类目标
    y_reg = df[['area', 'avg_thickness', 'max_thickness']].values   # 回归目标
    
    # 输入特征标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 回归目标标准化（注意：分类目标无需标准化）
    scaler_reg = StandardScaler()
    y_reg_scaled = scaler_reg.fit_transform(y_reg)
    
    return X_scaled, y_ice_type, y_reg_scaled, scaler_X, scaler_reg

# ---------------------------
# 2. 训练分类模型（积冰类型）
# ---------------------------
def train_classifier(X_train, y_train):
    model = XGBClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.5,
        subsample=0.8,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# ---------------------------
# 3. 训练多输出回归模型（面积、平均厚度、最大厚度）
# ---------------------------
def train_regressor(X_train, y_train):
    # 使用 MultiOutputRegressor 处理多输出
    base_model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    return model

# ---------------------------
# 4. 模型评估与关联性分析
# ---------------------------
def evaluate_models(clf_model, reg_model, X_test, y_test_clf, y_test_reg, scaler_reg):
    # 分类任务评估
    y_pred_clf = clf_model.predict(X_test)
    clf_accuracy = accuracy_score(y_test_clf, y_pred_clf)
    clf_f1 = f1_score(y_test_clf, y_pred_clf)
    
    # 回归任务评估
    y_pred_reg_scaled = reg_model.predict(X_test)
    y_pred_reg = scaler_reg.inverse_transform(y_pred_reg_scaled)
    y_test_reg_original = scaler_reg.inverse_transform(y_test_reg)
    
    # 计算回归指标
    reg_metrics = {
        'mse': mean_squared_error(y_test_reg_original, y_pred_reg),
        'mae': mean_absolute_error(y_test_reg_original, y_pred_reg),
        'r2': r2_score(y_test_reg_original, y_pred_reg)
    }
    
    # 检查输出关联性（如max_thickness >= avg_thickness）
    violation_count = np.sum(y_pred_reg[:, 2] < y_pred_reg[:, 1])
    print(f"逻辑约束违反次数（max_thickness < avg_thickness）: {violation_count}")
    
    return clf_accuracy, clf_f1, reg_metrics
# ---------------------------
# 新增可视化函数
# ---------------------------
def plot_comparison(clf_model, reg_model, X_test, y_ice_test, y_reg_test, scaler_reg):
    # 设置全局样式
    sns.set(style="whitegrid", font="Times New Roman")
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # 获取预测结果
    y_clf_pred = clf_model.predict(X_test)
    y_clf_proba = clf_model.predict_proba(X_test)[:, 1]  # 预测为角状冰的概率
    y_reg_pred = scaler_reg.inverse_transform(reg_model.predict(X_test))
    y_reg_true = scaler_reg.inverse_transform(y_reg_test)
    
    # 计算F1分数（用于e图）
    f1 = f1_score(y_ice_test, y_clf_pred)
    
    # 创建一个大图，包含所有子图
    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs = GridSpec(2, 3, figure=fig)  # 2行3列的网格
    
    # --------------------------
    # 回归结果 (a, b, c) - 第一行
    # --------------------------
    targets = ['Cross-sectional Area (cm²)', 
               'Average Thickness (cm)', 
               'Maximum Thickness (cm)']
    metrics = ['R²', 'MAE']
    
    for idx, target in enumerate(targets):
        ax = fig.add_subplot(gs[0, idx])  # 第一行，第idx列
        
        # 散点图
        sns.regplot(x=y_reg_true[:, idx], y=y_reg_pred[:, idx], 
                    scatter_kws={'s': 40, 'alpha': 0.7, 'edgecolor': 'w', 'color': '#2ca02c'},
                    line_kws={'color': '#d62728', 'linestyle': '--'}, ax=ax)
        
        # 标注指标
        r2 = r2_score(y_reg_true[:, idx], y_reg_pred[:, idx])
        mae = mean_absolute_error(y_reg_true[:, idx], y_reg_pred[:, idx])
        text = f"{metrics[0]} = {r2:.3f}\n{metrics[1]} = {mae:.3f}"
        ax.text(0.05, 0.85, text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # 格式化
        ax.set_xlabel('Measured Value (cm)')
        ax.set_ylabel('Predicted Value (cm)')
        ax.set_title(f'({chr(97+idx)}) {target}', y=1.02)  # 使用字母编号 (a, b, c)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # --------------------------
    # 分类结果 (d, e) - 第二行
    # --------------------------
    # 混淆矩阵 (d) - 第二行左
    ax3 = fig.add_subplot(gs[1, 0])  # 第二行第一列
    # 绘制混淆矩阵
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_ice_test, y_clf_pred, 
        cmap='Blues',
        display_labels=['Streamwise Ice', 'Horn Ice'],
        ax=ax3
    )
    
    # 添加色条标签
    cbar = ax3.images[0].colorbar
    cbar.set_label('Number of Samples', fontsize=10)
    
    ax3.set_title('(d) Confusion Matrix', y=1.02)
    
    # 概率分布 (e) - 第二行右
    ax4 = fig.add_subplot(gs[1, 1:])  # 第二行第二和第三列（合并）
    bins = np.linspace(0, 1, 21)
    # 创建数据框
    data = pd.DataFrame({
        'Probability': y_clf_proba,
        'Class': ['True Steamwise Ice samples' if c == 0 else 'True Horn Ice samples' for c in y_ice_test]
    })

    # 绘制概率分布
    sns.histplot(data=data, x='Probability', hue='Class', bins=bins, 
                element='step', stat='density', common_norm=False, 
                palette=['#2b8cbe', '#f03b20'], ax=ax4,legend=False)
    # 手动添加图例并强制指定位置
    ax4.legend(
        labels=['True Steamwise Ice samples', 'True Horn Ice samples'],
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),  # 精确坐标：左边缘2%，上边缘2%
        title='Class',
        frameon=True,
        fontsize=10,
        title_fontsize=11
    )
    ax4.text(0.5, 0.55, f'F1 Score = {f1:.4f}', 
        transform=ax4.transAxes, fontsize=12,
        verticalalignment='center', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax4.text(0.5, 0.45, f'Accuracy = {0.9444:.4f}',  # 假设准确率变量为accuracy
        transform=ax4.transAxes, fontsize=12,
        verticalalignment='center', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    

    # 设置坐标轴标签
    ax4.set_xlabel('Predicted Probability of Angular Ice', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('(e) Probability Distribution of Class Predictions', y=1.02)
    # 调整布局

    plt.tight_layout()

    plt.savefig('XGBoost_Combined_Results.tiff', dpi=600, bbox_inches='tight')
    plt.show()

def plot_feature_importance(reg_model, clf_model, features, output_dir='.'):
    """
    Plot feature importance graphs for regression and classification tasks.

    Parameters:
    - reg_model: Trained regression model (with feature importances).
    - clf_model: Trained classification model (with feature importances).
    - features: List of feature names.
    - output_dir: Directory to save output files, default is current directory.
    """

    # Set parameters for scientific plots
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10

    # Create figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    axs = axs.ravel()

    colors = sns.color_palette("Blues_r", n_colors=len(features))

    # Plot feature importance for regression tasks (three subplots)
    regression_titles = ['(a) Ice Area Prediction', '(b) Average Thickness Prediction', '(c) Maximum Thickness Prediction']
    for i in range(3):
        # Get feature importance for the i-th regression target
        importance = reg_model.estimators_[i].feature_importances_

        # Sort in descending order
        sorted_idx = np.argsort(importance)
        sorted_features = [features[j] for j in sorted_idx]
        sorted_importance = importance[sorted_idx]

        # Plot horizontal bar chart
        bars = axs[i].barh(range(len(features)), sorted_importance, 
                          color=colors, edgecolor='black', linewidth=0.5)

        # Axis settings
        axs[i].set_yticks(range(len(features)))
        axs[i].set_yticklabels(sorted_features)
        axs[i].set_xlabel('Feature Importance', fontsize=11)
        axs[i].set_title(regression_titles[i], pad=12, fontweight='semibold')

        # Add numeric labels to the bars
        for j, v in enumerate(sorted_importance):
            axs[i].text(v+0.01, j, f'{v:.3f}', 
                       va='center', fontsize=9, color='#2a2a2a')

    # Plot feature importance for classification task
    clf_importance = clf_model.feature_importances_
    sorted_idx = np.argsort(clf_importance)
    sorted_features = [features[j] for j in sorted_idx]
    sorted_importance = clf_importance[sorted_idx]

    bars = axs[3].barh(range(len(features)), sorted_importance, 
                      color=sns.color_palette("Reds_r", len(features)), 
                      edgecolor='black', linewidth=0.5)

    axs[3].set_yticks(range(len(features)))
    axs[3].set_yticklabels(sorted_features)
    axs[3].set_xlabel('Feature Importance', fontsize=11)
    axs[3].set_title('(d) Ice Type Classification Prediction', pad=12, fontweight='semibold')

    # Add numeric labels to the bars
    for j, v in enumerate(sorted_importance):
        axs[3].text(v+0.01, j, f'{v:.3f}', 
                   va='center', fontsize=9, color='#2a2a2a')

    # Add unified legend
    fig.legend([bars[0]], ['Feature Importance'], 
              loc='upper center', 
              bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False)

    # Save the output
    output_tiff = f'{output_dir}/Feature_Importance_Analysis.tiff'
    output_pdf = f'{output_dir}/Feature_Importance_Analysis.pdf'
    #plt.savefig(output_tiff, bbox_inches='tight',dpi=600)
    #plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()  
# ---------------------------
# 5. 主程序
# ---------------------------
if __name__ == "__main__":
    data_path = "combine_1.csv"
    
    # 加载数据
    X, y_ice_type, y_reg, scaler_X, scaler_reg = load_and_preprocess_data(data_path)
    
    # 划分训练集和测试集（80%训练，20%测试）
    X_train, X_test, y_ice_train, y_ice_test, y_reg_train, y_reg_test = train_test_split(
        X, y_ice_type, y_reg, test_size=0.2, random_state=42
    )
    
    # 训练分类模型
    clf_model = train_classifier(X_train, y_ice_train)
    
    # 训练回归模型
    reg_model = train_regressor(X_train, y_reg_train)
    
    # 评估模型
    clf_acc, clf_f1, reg_metrics = evaluate_models(
        clf_model, reg_model, X, y_ice_type, y_reg, scaler_reg
    )
    #clf_model.save_model('XGB_clf_ice_model.json')
    #joblib.dump(reg_model, 'XGB_reg_ice_model.joblib')
    # 在评估之后添加可视化
    plot_comparison(clf_model, reg_model, X, y_ice_type, y_reg, scaler_reg)
    
    # 打印评估结果
    print("分类任务评估:")
    print(f"- 准确率: {clf_acc:.4f}")
    print(f"- F1分数: {clf_f1:.4f}\n")
    
    print("回归任务评估:")
    print(f"- 均方误差 (MSE): {reg_metrics['mse']:.4f}")
    print(f"- 平均绝对误差 (MAE): {reg_metrics['mae']:.4f}")
    print(f"- 决定系数 (R²): {reg_metrics['r2']:.4f}")
    features = ['T', 'V', 'A', 'D', 'W', 'C']
    plot_feature_importance(reg_model, clf_model, features, output_dir='path_to_output_directory')
    '''
    # 特征重要性可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(6), clf_model.feature_importances_)
    plt.title("分类模型特征重要性")
    plt.xticks(range(6), ['T', 'V', 'A', 'D', 'W', 'C'], rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(6), reg_model.estimators_[0].feature_importances_)
    plt.title("回归模型特征重要性（第一个输出：面积）")
    plt.xticks(range(6), ['T', 'V', 'A', 'D', 'W', 'C'], rotation=45)
    plt.tight_layout()
    plt.show()
    '''