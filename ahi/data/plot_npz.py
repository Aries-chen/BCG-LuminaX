import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import pearsonr

def calculate_metrics(pred, label):
    """
    Calculate F1 score, Pearson correlation, Precision, and Recall.
    
    Args:
        pred (torch.Tensor): Predicted tensor with shape [batch_size, length].
        label (torch.Tensor): Ground truth tensor with shape [batch_size, length].
    
    Returns:
        dict: A dictionary containing F1, Pearson, Precision, and Recall.
    """
    assert pred.shape == label.shape, "Pred and label must have the same shape."
    
    # Convert to numpy arrays for compatibility with metrics functions
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    
    # Flatten tensors to 1D arrays
    pred_flat = pred.flatten()
    label_flat = label.flatten()
    
    # For F1, Precision, and Recall, threshold predictions at 0.5 (assuming binary classification)
    pred_binary = (pred_flat > 0.5).astype(int)
    label_binary = label_flat.astype(int)
    
    # F1 Score
    f1 = f1_score(label_binary, pred_binary, zero_division=0)
    
    # Precision
    precision = precision_score(label_binary, pred_binary, zero_division=0)
    
    # Recall
    recall = recall_score(label_binary, pred_binary, zero_division=0)
    
    # Pearson Correlation
    pearson_corr, _ = pearsonr(pred_flat, label_flat)
    
    return {
        "F1": f1,
        "Pearson": pearson_corr,
        "Precision": precision,
        "Recall": recall
    }

# 示例用法
if __name__ == "__main__":
    # 创建随机的示例数据
    batch_size = 256
    length = 300

    # 模拟预测和标签
    pred = torch.rand(batch_size, length)  # 预测值
    label = torch.randint(0, 2, (batch_size, length)).float()  # 标签 (0 或 1)

    metrics = calculate_metrics(pred, label)
    print(metrics)
