"""
FMATO 核心实验执行流水线 (run_experiments.py)
自动完成高频数据处理、特征提取、模型库训练、IRL多尺度权重求解、在线动态选择回测与指标保存。
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.config import DATASET_PATHS, RESULTS_DIR, SUMMARY_JSON_PATH, HORIZONS
from src.data_loader import (
    load_and_preprocess_ticks,
    extract_microstructure_features,
    prepare_train_test_split
)
from src.model_library import build_model_library, train_model_library
from src.irl_reward import IRLRewardLearner
from src.model_selector import build_all_selectors
from src.execution_engine import ExecutionEngine


def run_single_dataset_experiment(instrument_key: str, csv_path: str) -> Dict[str, Any]:
    """
    针对单个高频数据集执行完整的 FMATO 复现实验
    """
    print(f"\n=======================================================")
    print(f"🚀 开始执行数据集实验: [{instrument_key}] - {csv_path}")
    print(f"=======================================================")
    
    start_total_time = time.perf_counter()

    # 1. 加载盘口数据与特征工程
    raw_df = load_and_preprocess_ticks(csv_path)
    df_feat, feature_cols = extract_microstructure_features(raw_df)
    split_info = prepare_train_test_split(df_feat, feature_cols)
    
    train_df = split_info["train_df"]
    test_df = split_info["test_df"]
    X_train = split_info["X_train"]
    X_test = split_info["X_test"]
    y_train = split_info["y_train"]
    y_test = split_info["y_test"]

    # 2. 采集盘口切片数据 (用于可视化看板展示 L2 深度价量)
    l2_snapshot = []
    sample_rows = test_df.iloc[:5]
    for _, row in sample_rows.iterrows():
        l2_snapshot.append({
            "timestamp": str(row["ts_event"]),
            "bids": [{"price": float(row[f"bid_px_0{i}"]), "size": float(row[f"bid_sz_0{i}"])} for i in range(5)],
            "asks": [{"price": float(row[f"ask_px_0{i}"]), "size": float(row[f"ask_sz_0{i}"])} for i in range(5)],
            "spread": float(row["spread"]),
            "obi_l1": float(row["obi_l1"]),
        })

    # 3. 构建并训练轻量模型库
    models = build_model_library()
    latencies = train_model_library(models, X_train, y_train)

    # 评估各单体模型在测试集上的预测能力
    model_eval_results = []
    all_test_preds = np.zeros((len(test_df), len(models)))
    for idx, model in enumerate(models):
        preds = model.predict(X_test)
        all_test_preds[:, idx] = preds
        mse = float(np.mean((preds - y_test) ** 2))
        dir_acc = float(np.mean(np.sign(preds) == np.sign(y_test)))
        model_eval_results.append({
            "name": model.name,
            "mse": round(mse * 1e6, 4), # 乘以 1e6 便于阅读
            "direction_accuracy": round(dir_acc * 100, 2), # 百分比
            "latency_us": round(latencies[model.name], 2),
        })
        print(f"  - [{model.name}] 方向准确率: {dir_acc*100:.2f}%, MSE(e-6): {mse*1e6:.4f}")

    # 4. 逆强化学习 (IRL) 求解多尺度奖励权重 w*
    irl_learner = IRLRewardLearner(horizons=HORIZONS)
    irl_weights = irl_learner.fit_reward_weights(train_df)
    weights_dict = {f"{h}_ticks": round(float(w), 4) for h, w in zip(HORIZONS, irl_weights)}

    # 5. 执行各策略在线选择回测
    print(f"\n[Execution] 开始运行 7 大策略在线动态回测...")
    selectors = build_all_selectors(models)
    engine = ExecutionEngine(instrument_key=instrument_key, irl_learner=irl_learner)

    strategy_results = []
    for sel in selectors:
        res = engine.run_backtest(
            selector=sel,
            test_df=test_df,
            X_test=X_test,
            all_model_preds=all_test_preds
        )
        strategy_results.append(res)
        print(f"  > [{res['strategy']:<24}] 成交: {res['total_trades']:4d}笔 | "
              f"胜率: {res['win_rate']*100:5.2f}% | 毛收益: {res['gross_return']*100:6.2f}% | "
              f"净收益: {res['net_return']*100:6.2f}% | 夏普: {res['sharpe_ratio']:5.2f}")

    elapsed_time = round(time.perf_counter() - start_total_time, 2)
    print(f"✔ 数据集 [{instrument_key}] 实验完成 (总耗时: {elapsed_time}s)")

    return {
        "instrument": instrument_key,
        "sample_size": len(df_feat),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "features": feature_cols,
        "elapsed_seconds": elapsed_time,
        "l2_snapshot": l2_snapshot,
        "model_eval": model_eval_results,
        "irl_weights": weights_dict,
        "strategies": strategy_results,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "horizons": HORIZONS,
        "experiments": {}
    }

    # 对配置中的两个高频数据集依次执行
    for key, path in DATASET_PATHS.items():
        if os.path.exists(path):
            exp_res = run_single_dataset_experiment(key, path)
            summary_output["experiments"][key] = exp_res
        else:
            print(f"⚠ 警告: 未找到数据文件 {path}，跳过该项。")

    # 保存结构化结果 JSON
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, ensure_ascii=False, indent=2)

    print(f"\n=======================================================")
    print(f"🎉 全部实验执行完毕！结果已汇总保存至: {SUMMARY_JSON_PATH}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
