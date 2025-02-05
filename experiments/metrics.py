import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 真实数据
actual_prices = [1.763, 1.766, 1.739, 1.772, 1.783, 1.776]
actual_directions = np.sign(np.diff(actual_prices)).astype(int)  # [1, -1, 1, 1, -1]

# 预测数据集合
models = {
    # Base Models
    "DeepSeek": [1.758, 1.768, 1.763, 1.732, 1.782, 1.788],
    "ChatGPT 4o": [1.751, 1.756, 1.764, 1.755, 1.762, 1.767],
    "Doubao": [1.741, 1.739, 1.761, 1.752, 1.758, 1.760],
    "Kimi": [1.760, 1.760, 1.765, 1.740, 1.770, 1.780],
    "Claude": [1.770, 1.771, 1.762, 1.745, 1.783, 1.791],
    "LSTM": [1.836, 1.831, 1.832, 1.835, 1.839, 1.826],
    
    # Models with Sentiment
    "DeepSeek+Sentiment": [1.758, 1.768, 1.762, 1.748, 1.785, 1.795],
    "ChatGPT+Sentiment": [1.765, 1.766, 1.770, 1.735, 1.780, 1.790],
    "Doubao+Sentiment": [1.760, 1.770, 1.770, 1.745, 1.780, 1.790],
    "Kimi+Sentiment": [1.770, 1.775, 1.770, 1.755, 1.780, 1.790],
    "Claude+Sentiment": [1.768, 1.769, 1.770, 1.745, 1.779, 1.777],
    "LSTM+Sentiment": [1.833, 1.831, 1.831, 1.828, 1.832, 1.831],
    
    # Our Model
    "Ours": [1.762, 1.770, 1.763, 1.758, 1.785, 1.796]
}

# ================= 方向指标计算 =================
def calculate_direction_metrics(name, pred_prices):
    # 生成预测方向序列（与前一交易日比较）
    pred_directions = []
    for i in range(1, 6):  # 只计算1月21日到27日的5次变化
        pred_directions.append(1 if pred_prices[i] > actual_prices[i-1] else -1)
    
    # 计算指标
    return {
        "Accuracy": accuracy_score(actual_directions, pred_directions),
        "Precision": precision_score(actual_directions, pred_directions, average='macro'),
        "Recall": recall_score(actual_directions, pred_directions, average='macro'),
        "F1": f1_score(actual_directions, pred_directions, average='macro')
    }

# ================= 连续指标计算 =================
def calculate_continuous_metrics(name, pred_prices):
    # 注意：预测值需要对应实际值的日期（1月20日到27日共6个预测）
    errors = [abs(p - a) for p, a in zip(pred_prices, actual_prices)]
    return {
        "MAE": np.mean(errors),
        "RMSE": np.sqrt(np.mean(np.square(errors))),
        "MAPE": np.mean([abs(e/a)*100 for e, a in zip(errors, actual_prices)])
    }

# ================= 执行计算 =================
direction_results = {}
continuous_results = {}

for model_name, preds in models.items():
    direction_results[model_name] = calculate_direction_metrics(model_name, preds)
    continuous_results[model_name] = calculate_continuous_metrics(model_name, preds)

# ================= 结果输出 =================
print("方向预测指标验证：")
for model, metrics in direction_results.items():
    print(f"{model:20} | Accuracy: {metrics['Accuracy']:.2f} | Precision: {metrics['Precision']:.2f} | Recall: {metrics['Recall']:.2f} | F1: {metrics['F1']:.2f}")

print("\n连续预测指标验证：")
for model, metrics in continuous_results.items():
    print(f"{model:20} | MAE: {metrics['MAE']:.3f} | RMSE: {metrics['RMSE']:.3f} | MAPE: {metrics['MAPE']:.2f}%")
