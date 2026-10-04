"""真实时间 ME 的有限实现，复用原期间选择器与美元执行器。

依据论文物理第4页 Eq.(2)–(4)、第5页 §3.3.1 和 Algorithms 2–3。
原文未公开 ME 聚合、极端信号筛选与长奖励成熟衔接。项目假设为：使用既有
交易阈值筛选非零方向预测，方向乘原始中间价差；在线期间/历史窗全部信号
都成熟才取均值，不用成熟子集替换分母。该均值不是 Eq.(5) 的订单均值。

ME 与账户成交无关，因此可以先因果推进预测反馈/选择，再将逐时刻选择送入
已有 TimeExecutionEngine。这个分解不新增成交模型；所有美元成本、延迟、
风险退出、持仓归属和换版处理仍由原执行器负责。
"""
from collections import Counter, deque

import numpy as np
import pandas as pd

from src.history_ars import HistoryARSSelector, HistoryWindowBacktester
from src.model_selector import SingleModelSelector
from src.period_ucb import PeriodOESelector, PeriodOrderBook
from src.snapshot_dataset import build_session_dataset, utc_ns
from src.time_execution import TimeExecutionEngine
from src.timed_snapshots import SNAPSHOT_SCHEMA
from src.weekly_library import predict_library


def prediction_rows(frame, calendar, interval_ms, predictions):
    """返回当下可决策行；仅依赖特征、当前预测和已知日历，不读未来标签。

    predictions 为 N×K 相对收益预测；动态选择需 K 列都有限。关闭边界与
    最后退出网格不发新预测意图，和既有完整 session 执行器完全相同。
    """
    times = utc_ns(frame.ts_event)
    step = interval_ms * 1_000_000
    values = np.asarray(predictions, float)
    if (len(frame) < 2 or frame.ts_event.dt.tz is None or (np.diff(times) <= 0).any()
            or (times % step).any() or values.ndim != 2 or values.shape[0] != len(frame)):
        raise ValueError('Grid timestamps and N by K causal predictions required')
    ends = dict(zip(calendar.segments.segment_id, calendar.ends))
    closes = np.array([ends[s] for s in frame.segment_id])
    return frame.feature_valid.to_numpy() & np.isfinite(values).all(axis=1) & (times + step < closes)


def signal_features(frame, origins, sides, reward, interval_ms, known_through_ns):
    """计算 S×H 预测时刻价格差与 S 个状态，明确限制观察上限。

    origins 为当前已产生的信号行号，sides 为±1；H 个毫秒前瞻全部成熟后才
    允许评价。精确目标缺格或跨连续段均失效。返回 NaN 的无效向量不会被送入
    求解器/选择器；这里不用 prepared 的离线标签或 learning_ready。
    """
    times = utc_ns(frame.ts_event)
    origins = np.asarray(origins, dtype=int); sides = np.asarray(sides, dtype=int)
    if (origins.shape != sides.shape or (origins < 0).any() or (origins >= len(frame)).any()
            or not np.isin(sides, [-1, 1]).all()):
        raise ValueError('Valid prediction origins and nonzero directions required')
    hs = np.asarray(reward.horizons, dtype=np.int64) * 1_000_000
    targets = times[origins, None] + hs
    stop = np.searchsorted(times, int(known_through_ns), side='right')
    if stop == 0:
        raise ValueError('Observation clock precedes all quotes')
    indices = np.searchsorted(times[:stop], targets)
    safe = np.minimum(indices, stop - 1)
    pending = targets[:, -1] > known_through_ns
    missing = ((indices >= stop) | (times[safe] != targets)).any(axis=1)
    blocks = np.cumsum(np.r_[False, (np.diff(times) != interval_ms * 1_000_000)
        | (frame.segment_id.to_numpy()[1:] != frame.segment_id.to_numpy()[:-1])
        | (frame.session_id.to_numpy()[1:] != frame.session_id.to_numpy()[:-1])])
    crossed = (blocks[safe] != blocks[origins, None]).any(axis=1)
    statuses = np.where(pending, 'pending_signals', np.where(missing, 'missing_target',
        np.where(crossed, 'crossed_gap_or_session', 'matured')))
    features = np.full((len(origins), len(hs)), np.nan)
    good = statuses == 'matured'
    mids = frame.mid_price.to_numpy(float)
    # Eq.(2) 中间价差单位为价格点；卖方向乘-1是项目扩展，不扣任何美元成本。
    features[good] = sides[good, None] * (mids[safe[good]] - mids[origins[good], None])
    return features, statuses


def calibration_me(frame, predictions, calendar, reward, interval_ms, threshold):
    """校准仅合并成熟的过阈值预测，零信号及坏目标不补零。

    输入 N×1 静态候选预测。成熟子集用来近似 Algorithm 1 的特征期望，
    与在线期间全部信号的均值总体不同；必须分别标记，不能混称 Eq.(5)。
    """
    values = np.asarray(predictions, float).reshape(len(frame), 1)
    allowed = prediction_rows(frame, calendar, interval_ms, values)
    direction = (values[:, 0] > threshold).astype(int) - (values[:, 0] < -threshold).astype(int)
    origins = np.flatnonzero(allowed & (direction != 0))
    features, statuses = signal_features(frame, origins, direction[origins], reward, interval_ms,
                                        int(utc_ns(frame.ts_event)[-1]))
    good = statuses == 'matured'; count = int(good.sum())
    return dict(prediction_signal_count=len(origins), matured_signal_count=count,
        prediction_feature_expectation=features[good].mean(axis=0).tolist() if count else None,
        signal_status_counts=dict(Counter(statuses)),
        sample_policy='abs_prediction_above_existing_trade_threshold',
        aggregation='calibration_matured_signal_subset_not_online_full_period')


class PredictionHistoryBacktester(HistoryWindowBacktester):
    """复用历史窗参数与可见版本检查，ME 直接评价预测，不模拟虚构订单。

    窗口内重新预热特征；所有过阈值信号都是分母。严格最近窗通常没有完整
    长奖励，这种缺失不通过缩短前瞻或丢弃未成熟信号来消除。
    """
    def evaluate(self, now, session, version_id, window_ms, alignment, owners, *, detail=False):
        """返回其他臂的完整窗口 ME；原始容器可含未来，实际取用严格止于 now。"""
        if (type(window_ms) is not int or window_ms <= 0 or window_ms % self.interval_ms
                or alignment not in ('recent', 'matured') or not owners
                or len(set(owners)) != len(owners)
                or any(type(i) is not int or not 0 <= i < len(self.model_ids) for i in owners)):
            raise ValueError('Grid-aligned history and valid owners required')
        longest = max(self.reward.horizons) * 1_000_000
        end = int(now) - (longest if alignment == 'matured' else 0)
        start = end - window_ms * 1_000_000
        metadata = dict(window_start_ns=start, window_end_ns=end, known_through_ns=int(now),
            reward_observation_latest_required_ns=end-self.step+longest, alignment=alignment,
            model_version_id=version_id, session_id=session, source='independent_prediction_history',
            feature_history='rebuilt_inside_window_only', reward_type='ME')
        visible = [v for v in self.versions if pd.Timestamp(v['available_at_utc']).value <= now]
        if not visible or visible[-1]['version_sha256'] != version_id:
            raise ValueError('Latest visible model version required for ME history')
        version = visible[-1]; info = self.calendar.sessions[session]
        reason = ('insufficient_session_history' if start < info['open'].value or end > info['close'].value
                  else 'window_before_current_version' if start < pd.Timestamp(version['available_at_utc']).value
                  else None)
        def failed(status):
            return {i: metadata | dict(owner=i, status=status, score=None, total_signals=None) for i in owners}
        if reason:
            return failed(reason)
        stop = int(np.searchsorted(self.times, now, side='right'))
        lo = int(np.searchsorted(self.times[:stop], start)); hi = int(np.searchsorted(self.times[:stop], end))
        if hi-lo < 2:
            return failed('insufficient_quotes')
        raw = self.frame.iloc[lo:hi][SNAPSHOT_SCHEMA.names+['session_id', 'segment_id']].copy()
        received = utc_ns(raw.source_ts_recv)
        if raw.source_ts_recv.isna().any() or (received >= self.times[lo:hi]).any():
            raise ValueError('ME historical quotes must already be received')
        assigned, outside, preopen = self.calendar.assign(raw)
        if (outside or preopen or not assigned.session_id.equals(raw.session_id)
                or not assigned.segment_id.equals(raw.segment_id)):
            raise ValueError('ME history disagrees with calendar')
        historical, _, _ = build_session_dataset(raw, self.calendar, self.interval_ms, (self.interval_ms,))
        matrix, _ = predict_library([version], historical, self.interval_ms)
        full_start = self.times[lo] == start or start == info['open'].value and self.times[lo] == start+self.step
        full_end = self.times[hi-1] == end-self.step
        # 观察仅需窗起点以后的已知价格；共享无方向价差，避免对每个臂重扫整日。
        observed = self.frame.iloc[lo:stop]; results = {}
        all_features, all_statuses = signal_features(observed, np.arange(hi-lo), np.ones(hi-lo, dtype=int),
            self.reward, self.interval_ms, int(now))
        for owner in owners:
            values = matrix[:, owner]
            allowed = prediction_rows(historical, self.calendar, self.interval_ms, values[:, None])
            direction = (values > self.threshold).astype(int) - (values < -self.threshold).astype(int)
            origins = np.flatnonzero(allowed & (direction != 0))
            features = all_features[origins]*direction[origins, None]
            statuses = all_statuses[origins]
            count = len(origins); matured = int((statuses == 'matured').sum())
            status = ('model_not_fitted' if version['models'][owner]['status'] != 'fitted'
                else 'incomplete_window_boundary' if not full_start or not full_end
                else 'no_signals' if not count else 'pending_signals' if 'pending_signals' in statuses
                else 'incomplete_signals' if matured != count else 'matured')
            mean = features.mean(axis=0) if status == 'matured' else None
            results[owner] = metadata | dict(owner=owner, status=status,
                score=self.reward.score(mean) if mean is not None else None, total_signals=count,
                matured_signal_count=matured, signal_status_counts=dict(Counter(statuses)),
                latest_signal_required_ns=int(self.times[lo+origins[-1]]+longest) if count else None,
                prediction_feature_expectation=mean.tolist() if mean is not None else None)
        return results


class ScheduledModelSelector(SingleModelSelector):
    """把已因果生成的 ME 选择送入原执行器，不让成交 OE 改写 ME 选择。

    只允许执行器消费同一个当前可决策时刻，不按行号压缩缺口。继承静态接口
    仅用于复用原交易器；每个时刻的模型实际来自独立期间 ME 的在线选择。
    """
    def __init__(self, name, models, schedule, origin_ns):
        super().__init__(name, models)
        self.reward_type = 'OE'
        self.schedule, self.origin_ns = schedule, int(origin_ns)

    def select_model(self, elapsed_ms):
        """按 UTC 纳秒精确匹配当前选择，不插值或提前使用下一条决策。"""
        return self.record(self.schedule[self.origin_ns + int(elapsed_ms)*1_000_000])


def replay_me(selector, frame, predictions, version_ids, reward, calendar, *, interval_ms=500,
              latency_ms=500, holding_review_ms=15000, threshold=.000015, detail=False):
    """先推进 ME 反馈/期间选择，再交原执行器按相同成本成交。

    每个预测信号保留原模型/版本归属，等待七尺度最长前瞻，完整期间全部
    信号解决后才更新 W。内部复用 PeriodOrderBook 的成熟队列机制；其计数
    代表信号，输出统一改为 signals，避免与真实成交数量混淆。
    """
    times = utc_ns(frame.ts_event); matrix = np.asarray(predictions, float)
    allowed = prediction_rows(frame, calendar, interval_ms, matrix)
    selector.reward_type = 'ME'
    book = PeriodOrderBook(selector, calendar, max(reward.horizons)*1_000_000)
    pending = deque(); schedule = {}; statuses = Counter()
    feature_sum = np.zeros(len(reward.horizons)); count = total = 0
    # 查询目标索引可预先建立，但反馈分数只在目标已到达的当前时刻读取。
    hs = np.asarray(reward.horizons, dtype=np.int64)*1_000_000
    targets = times[:, None]+hs; indices = np.searchsorted(times, targets)
    safe = np.minimum(indices, len(times)-1)
    blocks = np.cumsum(np.r_[False, (np.diff(times) != interval_ms*1_000_000)
        | (frame.segment_id.to_numpy()[1:] != frame.segment_id.to_numpy()[:-1])
        | (frame.session_id.to_numpy()[1:] != frame.session_id.to_numpy()[:-1])])
    mids = frame.mid_price.to_numpy(float); sessions = frame.session_id.to_numpy()
    for row, clock in enumerate(times):
        while pending and targets[pending[0][0], -1] <= clock:
            origin, side, key = pending.popleft()
            status = ('missing_target' if (indices[origin] > row).any()
                      or not np.array_equal(times[safe[origin]], targets[origin])
                      else 'crossed_gap_or_session' if (blocks[safe[origin]] != blocks[origin]).any()
                      else 'matured')
            values = side*(mids[safe[origin]]-mids[origin]) if status == 'matured' else None
            book.resolve(key, status, reward.score(values) if values is not None else None)
            statuses[status] += 1
            if values is not None:
                feature_sum += values; count += 1
                bucket = book.buckets[key]
                bucket['signal_feature_sum'] = (np.asarray(bucket.get('signal_feature_sum',
                    np.zeros(len(hs)))) + values).tolist()
        book.advance(int(clock))
        if allowed[row]:
            owner = book.choose(sessions[row], version_ids[row], int(clock))
            schedule[int(clock)] = owner
            direction = int(matrix[row, owner] > threshold)-int(matrix[row, owner] < -threshold)
            if direction:
                key = book.add_order(sessions[row], version_ids[row], owner, int(clock))
                pending.append((row, direction, key)); total += 1
    feedback = book.finish(int(times[-1]))
    for bucket in feedback:
        bucket['signals'] = bucket.pop('orders')
        bucket['status'] = {'no_orders': 'no_signals', 'pending_orders': 'pending_signals',
                            'incomplete_orders': 'incomplete_signals'}.get(bucket['status'], bucket['status'])
    statuses['unmatured_at_end'] = len(pending)
    quotes = frame[SNAPSHOT_SCHEMA.names+['session_id', 'segment_id', 'mid_price', 'feature_valid']].copy()
    quotes['feature_valid'] = frame.feature_valid.to_numpy() & np.isfinite(matrix).all(axis=1)
    proxy = ScheduledModelSelector(selector.name, selector.models, schedule, times[0])
    engine = TimeExecutionEngine(reward, interval_ms=interval_ms, calendar=calendar,
        latency_ms=latency_ms, holding_review_ms=holding_review_ms, threshold=threshold, force_replay_end=False)
    result = engine.run_backtest(proxy, quotes, matrix, detail=detail, model_version_ids=version_ids)
    if proxy.action_history != selector.action_history:
        raise RuntimeError('ME plan and execution consumed different decisions')
    result.update(reward_type='ME', reward_cost_mode='PaperME-no-cost',
        observed_rewards=len(selector.reward_history), prediction_signal_count=total,
        matured_signal_count=count, prediction_feature_expectation=(feature_sum/count).tolist() if count else None,
        selector_protocol='fixed_period_ME_'+selector.mode, selection_period_ms=selector.period_ms,
        exploration_c_price=selector.c, period_feedback=feedback, period_selections=selector.period_selections,
        period_status_counts=dict(Counter(p['status'] for p in feedback)),
        selector_version_statistics=selector.summary(),
        me_assumptions='thresholded_signed_predictions_full_signal_period_equal_period_W_no_top_quantile')
    result['reward_status']['ME'] = dict(statuses)
    # 原执行器还为静态代理计算辅助ME；其全决策计数包含零信号，不能混入
    # 本入口过阈值样本。逐版本ME以期间账本为准，美元/OE计数仍原样保存。
    result['model_version_audit'] = {v: {k: n for k, n in counts.items() if not k.startswith('ME_')}
        for v, counts in result['model_version_audit'].items()}
    result['model_version_audit_role'] = 'execution_OE_and_decisions_ME_in_period_ledger'
    if detail:
        result['reward_observations'] = []  # 执行器逐单 OE 不冒充 ME 期间反馈。
    if isinstance(selector, HistoryARSSelector):
        histories = [e for p in selector.period_selections for e in p['evaluations']
                     if e['source'] == 'independent_prediction_history']
        result.pop('exploration_c_price')
        result.update(history_alignment=selector.alignment, history_window_ms=selector.window_ms,
            history_status_counts=dict(Counter(e['status'] for e in histories)),
            scored_history_windows=sum(e['score'] is not None for e in histories),
            cold_start_periods=sum(p['cold_start'] for p in selector.period_selections))
    return result
