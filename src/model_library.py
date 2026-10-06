"""校准固定参照使用的 Ridge 适配器；主模型库由 weekly_library 构造。"""
from sklearn.linear_model import Ridge

class RidgeModel:
    """对已结束历史拟合线性相对收益预测；不从评价期选择超参数。"""
    def __init__(self, alpha=1.0):
        """alpha 为历史既定 L2 系数；训练标准化由调用方负责。"""
        self.name = 'Ridge_Linear'
        self.model = Ridge(alpha=alpha, random_state=42)
        self.is_trained = False

    def predict(self, features):
        """输入 N×F 已标准化特征，返回 N 个相对收益预测。"""
        return self.model.predict(features)
