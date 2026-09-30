"""课程实验总入口：将数据处理、奖励校准、验证调参与样本外回放串起来。

每个数据文件选取若干不重叠窗口，每个窗口独立执行四段流程：
1. 训练段拟合标准化、模型及可选奖励尺度；
2. 验证段静态集成选共同门槛，校准段按此门槛选专家、拟合奖励；
3. 验证段继续选择 UCB 系数和固定模型（存在重复选参风险）；
4. 测试段比较动态策略、基线和奖励消融，保存指标与可追溯元数据。

组员可先阅读 run_window 理解单窗实验，再阅读 main 理解多窗组织和结果保存。
测试收益只能用于最后报告，不得再反馈到前三段的选择中。"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from threadpoolctl import threadpool_limits
from src.config import (BASE_DIR, DATASET_PATHS, SAMPLE_EVENTS, HORIZON_EVENTS, SEED,
                        INITIAL_CAPITAL, QUANTITY, INSTRUMENT_CONFIG, RL_CONFIG,
                        UCB_CANDIDATES, SPLIT_RATIOS, SUMMARY_JSON_PATH)
from src.data_loader import (dataset_row_count, load_and_preprocess_events,
                             extract_microstructure_features, prepare_train_test_split)
from src.data_catalog import select_daily_source
from src.model_library import build_model_library, train_model_library
from src.irl_reward import IRLRewardLearner
from src.model_selector import (build_all_selectors, CausalEventUCBSelector, CausalShadowARSSelector,
                                SingleModelSelector, EnsembleSelector, FlatSelector, RandomSelector)
from src.execution_engine import ExecutionEngine
from src.artifacts import pretty_json


def file_hash(path):
    """分块计算文件 SHA-256，用于确认数据和源代码版本，而不是加密内容。

    每次最多读 1 MiB，避免为了计算大数据集哈希把整个文件加载到内存。"""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluate_models(models, predictions, y, train_y, threshold, X_test=None):
    """计算测试预测指标，同时给出可比较的简单基线。

    predictions 为 N×K，y 为长度 N 的真实未来收益；尾部没有标签的行只
    从预测评分中排除，执行引擎仍会使用这些行情。方向类别顺序统一为跌、平、涨。
    平衡准确率平均各个实际类别的召回率；非零方向准确率只看真实上涨/下跌样本，
    两者都不能与总体准确率混作同一个指标。多数类标签只在 train_y 上确定。
    返回模型指标列表和基线/类别分布字典，百分比字段使用 0～100 的数值。"""
    # 不能把没有未来价格的尾部 NaN 标签补成“价格不变”，否则会虚增平盘准确率。
    valid = np.isfinite(y)
    truth = np.sign(y[valid]).astype(int)
    # 训练多数类用于构造可部署的基线，而不是利用测试类别比例反选标签。
    majority = int(np.unique(np.sign(train_y), return_counts=True)[0][
        np.argmax(np.unique(np.sign(train_y), return_counts=True)[1])])
    rows = []
    for i, model in enumerate(models):
        p = predictions[valid, i]
        direction = np.sign(p).astype(int)
        nonzero = truth != 0
        trading = np.abs(p) > threshold
        threshold_direction = np.where(trading, direction, 0)
        rows.append(dict(name=model.name, mse=float(np.mean((p - y[valid]) ** 2)),
                         raw_sign_accuracy=float(np.mean(direction == truth) * 100),
                         balanced_accuracy=float(balanced_accuracy_score(truth, direction) * 100),
                         nonzero_direction_accuracy=float(np.mean(direction[nonzero] == truth[nonzero]) * 100)
                         if nonzero.any() else None,
                         confusion_matrix=confusion_matrix(truth, direction, labels=[-1, 0, 1]).tolist(),
                         latency_us=model.latency_us, latency=model.latency,
                         thresholded_signal_accuracy=float(np.mean(threshold_direction == truth) * 100),
                         active_signal_accuracy=float(np.mean(direction[trading] == truth[trading]) * 100)
                         if trading.any() else None,
                         signal_coverage=float(trading.mean()), threshold=float(threshold)))
        # 逻辑回归的类别 argmax 与期望收益符号不是同一预测，分别报告，避免误导。
        if hasattr(model, 'predict_class') and X_test is not None:
            classes = model.predict_class(X_test[valid])
            rows[-1]['classifier_argmax_accuracy'] = float(np.mean(classes == truth) * 100)
            rows[-1]['classifier_confusion_matrix'] = confusion_matrix(
                truth, classes, labels=[-1, 0, 1]).tolist()
    baselines = dict(evaluated_rows=int(valid.sum()), unlabeled_tail=int((~valid).sum()),
                     class_order=['down', 'flat', 'up'],
                     class_proportions={str(c): float(np.mean(truth == c)) for c in [-1, 0, 1]},
                     zero_prediction_mse=float(np.mean(y[valid] ** 2)),
                     zero_direction_accuracy=float(np.mean(truth == 0) * 100),
                     train_majority_class=majority,
                     train_majority_accuracy=float(np.mean(truth == majority) * 100))
    return rows, baselines


def static_selectors(models):
    """创建校准用的可执行候选策略：每个固定模型、等权集成与不交易。

    每次返回新实例，避免不同区间复用动作历史。现金也参与专家选择，
    防止所有交易都亏损时仍把最少亏损者宣传成优于不交易。"""
    return ([SingleModelSelector(f'Baseline-Single-{m.name}', models, i) for i, m in enumerate(models)]
            + [EnsembleSelector('Baseline-Static-Ensemble', models), FlatSelector('Baseline-Cash', models)])


def select_with_ties(values, candidates, tolerance=1e-9):
    """显式记录并列最优；按预先声明的候选顺序选择首项，不偷看测试表现。

    tolerance 是美元盈亏的绝对数值容差，不随利润大小放大。顺序来自配置，
    不能把并列时选择首个模型描述成统计意义上的唯一最优。
    """
    values = np.asarray(values, float)
    best = np.flatnonzero(np.abs(values - values.max()) <= tolerance)
    return dict(selected_index=int(best[0]), selected=candidates[int(best[0])],
                tied_best_candidates=[candidates[int(i)] for i in best],
                tie_break_rule='first_in_declared_candidate_order', tolerance_usd=tolerance)


def gated_me_expectation(learner, reference_prices, future_prices, signal, threshold):
    """只平均超过执行门槛的信号；零信号策略约定零特征，并返回有效信号数。

    这是论文强调极端预测的最小门槛化近似，未复刻 top-1/1000 排序协议。
    校准与测试采用同一门槛，不能校准所有微小预测、交易时却筛掉它们。
    """
    active = np.abs(signal) > threshold
    if not active.any():
        return np.zeros(len(learner.horizons)), 0
    features = learner.features_from_prices(reference_prices[active, None],
        future_prices[active], np.sign(signal[active, None]))
    return features.mean(axis=0), int(active.sum())


def run_window(key, path, offset, rows, seed, reward_definition="paper_price_difference",
               holding_period=RL_CONFIG['holding_period']):
    """完成一个原始数据窗口的独立实验，返回可序列化的详细结果。

    key 为品种配置键，path 为行情文件，offset/rows 按原始行定位窗口，
    seed 控制随机基线。每次都重新训练模型和初始化选择器，不继承其他窗口状态。
    输出包含真实时间范围、调参轨迹、奖励诊断、预测指标、策略对比和资金曲线。"""
    start = time.perf_counter()
    df, features = extract_microstructure_features(load_and_preprocess_events(path, rows, offset))
    split = prepare_train_test_split(df, features)
    models = build_model_library()
    train_model_library(models, split['X_train'], split['y_train'])
    # 模型此后冻结：预先计算整批预测只是提速，每行输入特征仍只依赖当时及过去行情。
    predictions = {part: np.column_stack([m.predict(split[f'X_{part}']) for m in models])
                   for part in ('calibration', 'validation', 'test')}
    base_reward = IRLRewardLearner(definition=reward_definition)
    base_reward.fit_scales(split['train_df'])
    base_threshold = INSTRUMENT_CONFIG[key]['trade_threshold']

    # 门槛选择不依赖奖励学习（静态集成不消费反馈），故先冻结验证门槛，再用同一
    # 门槛重放校准段。验证段反复选门槛/c/固定模型，有选择过拟合风险，报告须说明。
    # 先用静态集成选一个共同门槛，各策略共享，以减少比较中的执行差异。
    validation = split['validation_df']
    tuning = []
    for threshold in [base_threshold * x for x in (1, 2, 4)]:
        engine = ExecutionEngine(key, base_reward, threshold=threshold, holding_period=holding_period)
        result = engine.run_backtest(EnsembleSelector('validation', models), validation,
                                     all_model_preds=predictions['validation'])
        tuning.append(dict(threshold=threshold, net_pnl_usd=result['net_pnl_usd']))
    threshold_choice = select_with_ties([x['net_pnl_usd'] for x in tuning],
                                        [x['threshold'] for x in tuning])
    threshold = threshold_choice['selected']

    # 阶段二：只在校准段回放可执行策略；专家按实际扣费后的美元盈亏选取。
    calibration = split['calibration_df']
    cal_engine = ExecutionEngine(key, base_reward, threshold=threshold, holding_period=holding_period)
    cal_selectors = static_selectors(models)
    calibration_results = [cal_engine.run_backtest(s, calibration,
                           all_model_preds=predictions['calibration']) for s in cal_selectors]
    expert_choice = select_with_ties([r['net_pnl_usd'] for r in calibration_results],
                                   [s.name for s in cal_selectors])
    expert_idx = expert_choice['selected_index']
    names = [s.name for s in cal_selectors]
    rewards = {}
    cal_mid = calibration.mid_price.to_numpy()
    # 校准已经发生在测试之前，可以离线计算未来收益；这里只取能在校准段内部到期的信号。
    count = len(calibration) - max(HORIZON_EVENTS)
    future_prices = np.column_stack([cal_mid[h:h + count] for h in HORIZON_EVENTS])
    mu_me, me_counts = [], []
    for i, selector in enumerate(cal_selectors):
        signal = (np.zeros(count) if selector.flat else predictions['calibration'][:count].mean(axis=1)
                  if selector.ensemble else predictions['calibration'][:count, i])
        expectation, signals = gated_me_expectation(base_reward, cal_mid[:count], future_prices,
                                                    signal, threshold)
        mu_me.append(expectation)
        me_counts.append(signals)
    # ME 平均过门槛信号；OE 平均已成熟实际成交，分母及零样本情况必须分别保存。
    mu_oe = [r['order_feature_expectation'] for r in calibration_results]
    for reward_type, expectations in [('ME', mu_me), ('OE', mu_oe)]:
        learner = IRLRewardLearner(scales=base_reward.scales, definition=reward_definition)
        learner.fit_reward_weights(expectations, expert_idx, names)
        rewards[reward_type] = learner

    # 对 ME/OE 分别扫描探索系数，同时保存所有候选分数，便于核查选择依据。
    c_by_reward, c_trials, c_choices = {}, {}, {}
    for reward_type in ('ME', 'OE'):
        trials = []
        for c in UCB_CANDIDATES:
            engine = ExecutionEngine(key, rewards[reward_type], threshold=threshold, holding_period=holding_period)
            result = engine.run_backtest(CausalEventUCBSelector('validation', models, reward_type, c), validation,
                                         all_model_preds=predictions['validation'])
            trials.append(dict(c=c, net_pnl_usd=result['net_pnl_usd']))
        choice = select_with_ties([x['net_pnl_usd'] for x in trials], [x['c'] for x in trials])
        c_choices[reward_type] = choice
        c_by_reward[reward_type] = choice['selected']
        c_trials[reward_type] = trials
    # “最佳固定模型”也必须由验证段选出，不能事后挑测试表现最好的模型当基线。
    val_fixed = []
    for i, model in enumerate(models):
        result = ExecutionEngine(key, rewards['ME'], threshold=threshold, holding_period=holding_period).run_backtest(
            SingleModelSelector('validation', models, i), validation, all_model_preds=predictions['validation'])
        val_fixed.append(result['net_pnl_usd'])
    fixed_choice = select_with_ties(val_fixed, [m.name for m in models])
    best_fixed = fixed_choice['selected_index']

    # 阶段四：冻结以上全部选择，进入测试段。每个策略都使用独立账户和新选择器。
    test = split['test_df']
    selectors = build_all_selectors(models, c_by_reward, seed)
    selectors += [SingleModelSelector('Baseline-ValidationBest', models, best_fixed)]
    selectors += [RandomSelector(f'Baseline-Random-seed-{s}', models, s) for s in (seed + 1, seed + 2)]
    results = []
    for selector in selectors:
        result = ExecutionEngine(key, rewards[selector.reward_type], threshold=threshold, holding_period=holding_period).run_backtest(
            selector, test, all_model_preds=predictions['test'])
        results.append(result)
    # 消融实验：固定 ARS、门槛、撮合成本与训练尺度，只改变奖励权重。
    # 单位向量意味着仅使用一个时域；等权向量会在奖励学习器初始化时归一化。
    for reward_type in ('ME', 'OE'):
        for name, weights in [('Equal', np.ones(len(HORIZON_EVENTS)))] + [
                (f'Single-{h}', np.eye(len(HORIZON_EVENTS))[i]) for i, h in enumerate(HORIZON_EVENTS)]:
            learner = IRLRewardLearner(weights=weights, scales=base_reward.scales, definition=reward_definition)
            selector = CausalShadowARSSelector(f'Ablation-{reward_type}-CausalShadowARS-{name}', models, reward_type)
            results.append(ExecutionEngine(key, learner, threshold=threshold, holding_period=holding_period).run_backtest(
                selector, test, all_model_preds=predictions['test']))
    # signed_box 只放宽权重符号、仍保留 sum(w)=1；[-1,1] 是本项目正则边界，
    # 不能称作论文规定。NetOE 只更改成本项，保留主实验权重，以隔离该因素。
    sensitivity = {}
    for reward_type, expectations in [('ME', mu_me), ('OE', mu_oe)]:
        signed = IRLRewardLearner(scales=base_reward.scales, definition=reward_definition,
                                 weight_constraint='signed_box')
        signed.fit_reward_weights(expectations, expert_idx, names)
        sensitivity[reward_type] = dict(weights=signed.weights.tolist(), diagnostics=signed.diagnostics)
        selector = CausalShadowARSSelector(f'Sensitivity-{reward_type}-SignedBox', models, reward_type)
        results.append(ExecutionEngine(key, signed, threshold=threshold, holding_period=holding_period).run_backtest(
            selector, test, all_model_preds=predictions['test']))
    net_oe = IRLRewardLearner(weights=rewards['OE'].weights, scales=base_reward.scales,
                              definition=reward_definition, deduct_cost=True)
    results.append(ExecutionEngine(key, net_oe, threshold=threshold, holding_period=holding_period).run_backtest(
        CausalShadowARSSelector('Sensitivity-NetOE-FixedWeights', models, 'OE'), test,
        all_model_preds=predictions['test']))
    model_eval, prediction_baselines = evaluate_models(models, predictions['test'],
                                                       split['y_test'], split['y_train'], threshold, split['X_test'])
    # 报告实际墙钟时间，避免把不同活跃度的相同事件数误当成相同秒数。
    partitions = {}
    for part in ('train', 'calibration', 'validation', 'test'):
        frame = split[f'{part}_df']
        partitions[part] = dict(rows=len(frame), start=str(frame.ts_event.iloc[0]),
                                end=str(frame.ts_event.iloc[-1]),
                                duration_seconds=float((frame.ts_event.iloc[-1] - frame.ts_event.iloc[0]).total_seconds()))
    event_seconds = (df.ts_event.shift(-max(HORIZON_EVENTS)) - df.ts_event).dt.total_seconds().dropna()
    snapshot = test.iloc[0]
    l2_snapshot = dict(timestamp=str(snapshot.ts_event),
                       bids=[dict(price=float(snapshot[f'bid_px_{i:02d}']),
                                  size=float(snapshot[f'bid_sz_{i:02d}'])) for i in range(5)],
                       asks=[dict(price=float(snapshot[f'ask_px_{i:02d}']),
                                  size=float(snapshot[f'ask_sz_{i:02d}'])) for i in range(5)])
    return dict(instrument=key, source_offset=offset, requested_rows=rows, sample_size=len(df), seed=seed,
                holding_period_events=int(holding_period),
                l2_snapshot=l2_snapshot,
                symbol=str(df.symbol.iloc[0]), train_size=len(split['train_df']), test_size=len(test),
                partitions=partitions, purge_events=split['purge_events'], features=features,
                normalization=dict(mean=split['norm_mean'], std=split['norm_std']),
                horizon_90_seconds_quantiles={str(q): float(event_seconds.quantile(q)) for q in (.1, .5, .9)},
                model_eval=model_eval, prediction_baselines=prediction_baselines,
                reward_scales=base_reward.scales.tolist(), reward_definition=reward_definition,
                horizon_unit='events', model_library_design='heterogeneous_same_features_same_training_period',
                signed_weight_sensitivity=sensitivity, calibration_expert_choice=expert_choice,
                calibration_me_active_counts=me_counts,
                rewards={k: dict(weights=v.weights.tolist(), diagnostics=v.diagnostics) for k, v in rewards.items()},
                calibration_policies=[dict(strategy=r['strategy'], net_pnl_usd=r['net_pnl_usd'],
                                          matured_order_count=r['matured_order_count'],
                                          order_mean_defined=r['order_feature_expectation_defined'])
                                      for r in calibration_results],
                tuning=dict(threshold=threshold, threshold_trials=tuning, threshold_choice=threshold_choice,
                            fixed_choice=fixed_choice, c_choices=c_choices, c_by_reward=c_by_reward,
                            c_trials=c_trials, validation_best_model=models[best_fixed].name,
                            validation_fixed_pnl=val_fixed),
                strategies=results, elapsed_seconds=time.perf_counter() - start)


def aggregate(windows):
    """按同名策略汇总各窗口的美元净盈亏与平仓笔数。

    窗口起始资金各自重置，所以 sum 是独立回放盈亏之和，不是复利净值；
    窗口可能来自同一交易日，不把它们当独立日期去计算年化或显著性。"""
    names = [r['strategy'] for r in windows[0]['strategies']]
    output = []
    for name in names:
        rows = [next(r for r in w['strategies'] if r['strategy'] == name) for w in windows]
        pnl = [r['net_pnl_usd'] for r in rows]
        total_trades = sum(r['total_trades'] for r in rows)
        gross_sum = float(sum(r['gross_pnl_usd'] for r in rows))
        friction_sum = float(sum(r['friction_usd'] for r in rows))
        gross_wins = sum(r['gross_winning_trades'] for r in rows)
        net_wins = sum(r['net_winning_trades'] for r in rows)
        holding_sum = sum((r['holding_events_mean'] or 0.) * r['total_trades'] for r in rows)
        output.append(dict(strategy=name, windows=len(rows), net_pnl_usd_sum=float(sum(pnl)),
                           net_pnl_usd_mean=float(np.mean(pnl)), net_pnl_usd_min=float(min(pnl)),
                           net_pnl_usd_max=float(max(pnl)), total_trades=total_trades,
                           gross_pnl_usd_sum=gross_sum, friction_usd_sum=friction_sum,
                           avg_trade_gross_usd=(gross_sum / total_trades if total_trades else 0.),
                           avg_trade_friction_usd=(friction_sum / total_trades if total_trades else 0.),
                           gross_win_rate=(gross_wins / total_trades if total_trades else None),
                           net_win_rate=(net_wins / total_trades if total_trades else None),
                           holding_events_mean=(holding_sum / total_trades if total_trades else None)))
    return output


def main(argv=None):
    """解析命令行、固定计算线程、遍历数据窗口，并在全部成功后保存结果。

    --rows 控制每窗原始事件数，--windows 控制每品种窗口数，--seed 控制
    随机基线，--output 可让快速验证另存文件。元数据同时记录依赖与源文件哈希，
    其中源码哈希对应生成结果当时的文件；后续仅改注释不应伪造历史运行记录。
    argv 供自动化测试传入命令行参数；正常启动时使用进程实际参数。

    --data-index 与 --data-date 配对使用，只运行索引中选定日期的 ESZ5 文件。
    这一步仍然做单文件窗口实验，不能把结果称为跨日训练或论文长期回测。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=SAMPLE_EVENTS, help='Raw events per within-file window')
    parser.add_argument('--reward-definition', choices=['paper_price_difference', 'normalized_return'],
                        default='paper_price_difference', help='Reward units; selectors remain explicit event variants')
    parser.add_argument('--windows', type=int, default=3)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--holding-period', type=int, default=RL_CONFIG['holding_period'],
                        help='Minimum event interval between position reviews; all other settings stay fixed')
    parser.add_argument('--data-index', help='Databento index JSON; pair with --data-date')
    parser.add_argument('--data-date', help='UTC file date YYYY-MM-DD; not an exchange session')
    parser.add_argument('--include-degraded', action='store_true',
                        help='Explicitly allow a selected date marked degraded')
    parser.add_argument('--output', help='Output JSON; required for indexed daily input')
    args = parser.parse_args(argv)
    if args.rows < 2000 or args.windows < 1 or args.holding_period < 1:
        parser.error('--rows >= 2000, --windows >= 1 and --holding-period >= 1 are required')
    # 新输入必须显式声明日期和输出文件，避免以为已跑完三个月，或误覆盖旧结果。
    # 不传新参数时完全保留根目录 ES/Brent 的默认入口，方便复核历史实验。
    if bool(args.data_index) != bool(args.data_date):
        parser.error('--data-index and --data-date must be supplied together')
    if args.include_degraded and not args.data_index:
        parser.error('--include-degraded requires --data-index and --data-date')
    daily_source = None
    datasets = DATASET_PATHS
    if args.data_index:
        if not args.output:
            parser.error('Indexed daily input requires an explicit --output')
        try:
            daily_source = select_daily_source(args.data_index, args.data_date, args.include_degraded)
        except (ValueError, OSError, KeyError, TypeError) as error:
            parser.error(str(error))
        datasets = {'CME_ES': daily_source['source_file']}
        # 记录完整索引的哈希，并原样保留质量状态；显式允许并不意味着质量已修复。
        daily_source['index_sha256'] = file_hash(args.data_index)
        print(f"ESZ5: UTC file date {args.data_date}, condition={daily_source['condition']}; "
              'within-file windows only, not cross-day training', flush=True)
    # 保存产生结果时的源码快照哈希；即使工作区还未提交，也能标识实际运行内容。
    source_files = [BASE_DIR / 'run_experiments.py'] + sorted((BASE_DIR / 'src').glob('*.py'))
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE_DIR, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    summary = dict(schema_version=3, timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'), horizon_events=HORIZON_EVENTS,
                   metadata=dict(reward_definition=args.reward_definition, horizon_unit="events", seed=args.seed, rows=args.rows, windows=args.windows,
                                 python=platform.python_version(), platform=platform.platform(),
                                 packages={p: importlib.metadata.version(p) for p in
                                           ['numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow', 'threadpoolctl']},
                                 source_sha256={str(p.relative_to(BASE_DIR)): file_hash(p) for p in source_files},
                                 git_base_revision=revision, threads=1, initial_capital=INITIAL_CAPITAL,
                                 quantity=QUANTITY, split_ratios=SPLIT_RATIOS,
                                 execution=dict(RL_CONFIG, holding_period=args.holding_period),
                                 instruments=INSTRUMENT_CONFIG, ucb_candidates=UCB_CANDIDATES,
                                 scope='Disjoint windows within supplied files; not independent trading days.'),
                   experiments={})
    if daily_source:
        summary['metadata']['daily_source'] = daily_source
    # 限制 BLAS/OpenMP 等计算库线程，减少资源差异；耗时仍受硬件和系统负载影响。
    with threadpool_limits(limits=1):
        for key, path in datasets.items():
            if not Path(path).exists():
                raise FileNotFoundError(path)
            total = dataset_row_count(path)
            if total < args.rows * args.windows:
                raise ValueError(f'{key}: insufficient rows for nonoverlapping windows')
            # 在全文件范围均匀定位窗口；前面的总行数检查保证这些窗口互不重叠。
            offsets = np.linspace(0, total - args.rows, args.windows, dtype=int).tolist()
            windows = []
            for i, offset in enumerate(offsets):
                print(f'{key}: window {i + 1}/{args.windows}, offset={offset}', flush=True)
                window = run_window(key, path, offset, args.rows, args.seed, args.reward_definition,
                                    args.holding_period)
                if daily_source and window['symbol'] != daily_source['symbol']:
                    raise ValueError('Loaded window symbol disagrees with indexed ESZ5 contract')
                windows.append(window)
            summary['experiments'][key] = dict(source_file=Path(path).name, source_sha256=file_hash(path),
                                               source_rows=total, windows=windows, aggregate=aggregate(windows))
    out = Path(args.output or SUMMARY_JSON_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 先完整写临时文件再替换目标，防止中途失败留下半份 JSON 或覆盖有效结果。
    # 序列化拒绝 NaN/Infinity，指标无法定义时应明确使用 None/null。
    temporary = out.with_suffix('.tmp')
    temporary.write_text(pretty_json(summary) + '\n')
    temporary.replace(out)
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()
