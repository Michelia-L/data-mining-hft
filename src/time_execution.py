"""定时快照的真实时间回放，先支持固定模型、静态集成与现金基线。

论文物理第 3 页 §3.1 定义定时 tick，第 4 页 Eq.(2)–(4) 从预测/成交
时刻计算未来中间价差。本模块用 UTC 纳秒驱动、用毫秒配置；不把缺格后的
下一行当成目标价格。固定期间 OE-UCB 通过独立期间协议接入；不复用事件窗口。
"""
from collections import Counter, deque

import numpy as np
import pandas as pd

from src.config import INITIAL_CAPITAL, INSTRUMENT_CONFIG, QUANTITY
from src.account import Account
from src.model_selector import SingleModelSelector
from src.period_ucb import PeriodOESelector, PeriodOrderBook
from src.snapshot_dataset import SessionCalendar, utc_ns


def timestamp(value):
    """审计记录使用带 UTC 时区的字符串，避免把纳秒整数误解成事件号。"""
    return pd.Timestamp(int(value), unit='ns', tz='UTC').isoformat()


class TimeAccount(Account):
    """复用经过验证的美元账本；内部复核时钟为纳秒，输出持仓时间为毫秒。

父类只通过两个时钟相减检查复核期，调用方统一传入纳秒。
每笔成交及完整交易的输出移除 events/step 字段，以免混淆时间单位。
"""
    def fill(self, clock, side, owner, mid, bid, ask, opening):
        """执行价格、滑点、手续费与旧账本相同，仅明确成交的真实时间。"""
        order = super().fill(clock, side, owner, mid, bid, ask, opening)
        order['timestamp_ns'] = int(order.pop('step'))
        order['ts_event'] = timestamp(clock)
        return order

    def close(self, clock, mid, bid, ask):
        """平仓盈亏仍属于原开仓模型；风险退出不伪装为新模型的交易贡献。"""
        order = super().close(clock, mid, bid, ask)
        trade = self.trades[-1]
        trade['entry_time'] = timestamp(trade.pop('entry_step'))
        trade['exit_time'] = timestamp(trade.pop('exit_step'))
        trade['holding_ms'] = float(trade.pop('holding_events') / 1_000_000)
        return order


class TimeExecutionEngine:
    """只读取当时行情和冻结预测的时间回放器；不消费任何离线未来标签。

固定选择器也记录成熟反馈，便于验证 ME/OE 的观察时点，但不会据此换模型。
期间选择器只能接收完整期间的 OE 均值；旧事件选择器仍拒绝直接接入。
"""
    def __init__(self, reward, *, interval_ms=500, latency_ms=500,
                 holding_review_ms=15000, calendar=None,
                 initial_capital=INITIAL_CAPITAL, quantity=QUANTITY, threshold=None,
                 force_replay_end=True):
        """半秒网格、半秒延迟、15 秒复核均为项目设定，不是论文公开的执行参数。

延迟可以不是网格倍数：使用到期后首条可用且连续的行情，不提前插值成交。
奖励前瞻期必须是网格倍数；本入口用 price_difference 和固定权重隔离时间改动。
force_replay_end=True 保留短窗口实验的末行平仓；完整 session 协议设为 False，
只按已知日历或当时可见的风险退出，缺少尾部行情时保留未平仓风险。
"""
        settings = (interval_ms, latency_ms, holding_review_ms)
        horizons = tuple(reward.horizons)
        if (any(type(x) is not int or x <= 0 for x in settings)
                or not horizons or any(type(h) is not int or h <= 0 or h % interval_ms for h in horizons)
                or tuple(sorted(set(horizons))) != horizons
                or horizons[-1] > 7 * 86400000):
            raise ValueError('Positive millisecond settings and increasing grid-aligned horizons required')
        if reward.definition != 'paper_price_difference' or reward.deduct_cost:
            raise ValueError('Time baseline requires price-difference reward without cost deduction')
        self.config = INSTRUMENT_CONFIG['CME_ES']
        self.threshold = self.config['trade_threshold'] if threshold is None else threshold
        if (not np.isfinite([initial_capital, quantity, self.threshold]).all()
                or initial_capital <= 0 or quantity <= 0 or self.threshold < 0):
            raise ValueError('Invalid account or threshold settings')
        self.reward, self.calendar = reward, calendar or SessionCalendar()
        if type(force_replay_end) is not bool:
            raise ValueError('force_replay_end must be boolean')
        # 小窗口保留旧期末退出假设；完整 session 协议禁用，避免缺尾时回溯强平。
        self.force_replay_end = force_replay_end
        self.interval_ms, self.latency_ms = interval_ms, latency_ms
        self.holding_review_ms = holding_review_ms
        self.initial_capital, self.quantity = initial_capital, quantity

    def run_backtest(self, selector, frame, predictions, *, detail=False, model_version_ids=None):
        """每行执行到期意图、结算成熟奖励、盯市，再依据当前有效特征发新意图。

意外缺口只能在下一条行情到达时发现：撤销旧意图，在这条行情上风险平仓，
保留缺口期间的价格损益。禁止偷看下一行并在缺口前回溯平仓。
计划暂停/收盘在日历已知的最后一个严格早于边界的网格主动退出；边界快照
只用于盯市和奖励，不假定边界仍能成交。退出行情缺失时，持仓延续到首条
可交易行情再处理。期末强平也是明确的离线工程假设，不能当作实盘撮合。

可选 model_version_ids 是 N 个当前可见库版本 ID；不提供时保留旧基线口径。
换版撤销旧待成交意图，已经成交的仓位/奖励仍属于原开仓版本。未平仓资产
不能因换版消失，旧反馈不能改标为新版本。静态选择器仍不按反馈换候选。
"""
        periodic = isinstance(selector, PeriodOESelector)
        if not isinstance(selector, (SingleModelSelector, PeriodOESelector)) or selector.needs_shadow:
            raise ValueError('Time replay supports static or explicit period OE selectors only')
        if selector.action_history or selector.reward_history:
            raise ValueError('Use a fresh selector for each replay')
        if periodic and (selector.period_selections or selector.period_ms % self.interval_ms):
            raise ValueError('Fresh selector and grid-aligned period required')
        frame = frame.reset_index(drop=True)
        n = len(frame)
        if n < 2:
            raise ValueError('At least two snapshots are required')
        times = utc_ns(frame.ts_event)
        step = self.interval_ms * 1_000_000
        if (frame.ts_event.dt.tz is None or (np.diff(times) <= 0).any() or (times % step).any()):
            raise ValueError('Strictly increasing timezone-aware grid timestamps required')
        received = utc_ns(frame.source_ts_recv)
        if (frame.source_ts_recv.dt.tz is None or frame.source_ts_recv.isna().any()
                or (received >= times).any()):
            raise ValueError('Replay quotes must have arrived strictly before their snapshot')
        assigned, outside, preopen = self.calendar.assign(frame)
        if (outside or preopen or not assigned.session_id.equals(frame.session_id)
                or not assigned.segment_id.equals(frame.segment_id)):
            raise ValueError('Replay session metadata disagrees with calendar')
        mid, bid, ask = (frame[c].to_numpy(float) for c in ('mid_price', 'bid_px_00', 'ask_px_00'))
        if (not np.isfinite(np.c_[mid, bid, ask]).all() or (bid <= 0).any()
                or (ask < bid).any() or not np.allclose(mid, (bid + ask) / 2, rtol=0, atol=1e-9)):
            raise ValueError('Invalid replay quotes')
        if frame.feature_valid.dtype != bool:
            raise ValueError('Feature availability must be boolean')
        available = frame.feature_valid.to_numpy()
        predictions = np.asarray(predictions, float)
        if predictions.shape != (n, selector.k) or not np.isfinite(predictions[available]).all():
            raise ValueError('Finite causal predictions aligned to feature-valid rows required')
        versions = None
        if model_version_ids is not None:
            versions = np.asarray(model_version_ids, dtype=object)
            if (versions.shape != (n,) or any(v is not None and (not isinstance(v, str) or not v) for v in versions)
                    or any(v is None for v in versions[available])):
                raise ValueError('Visible version IDs required on feature-valid rows')
        sessions, segments = frame.session_id.to_numpy(), frame.segment_id.to_numpy()
        changed = np.r_[False, (np.diff(times) != step) | (sessions[1:] != sessions[:-1])
                         | (segments[1:] != segments[:-1])]
        # block 仅由当前和过去时刻推导，绝不根据未来 learning_ready 决定能否交易。
        blocks = np.cumsum(changed)
        ends = dict(zip(self.calendar.segments.segment_id, self.calendar.ends))
        account = TimeAccount(self.config, self.holding_review_ms * 1_000_000, self.quantity)
        intents, orders, signals = deque(), deque(), deque()
        hs = np.array(self.reward.horizons, dtype=np.int64) * 1_000_000
        longest = int(hs[-1])
        periods = PeriodOrderBook(selector, self.calendar, longest) if periodic else None
        feature_sum = np.zeros(len(hs))
        statuses = {'ME': Counter(), 'OE': Counter()}
        exits, cancelled = Counter(), Counter()
        observations, decisions = [], []
        gross_values, net_values = [], []
        matured_orders = 0
        position_version = None
        version_audit = {}

        def audit_version(version, name):
            """版本级计数只在显式提供身份时开启，不改变默认结果 schema。"""
            if versions is not None:
                version_audit.setdefault(version, Counter())[name] += 1

        def mature(queue, now, kind):
            """仅在最长前瞻到期后检查已到达行情；缺格奖励不补零、不送入选择器。

            即便索引容器包含整段历史，实际取出的目标索引严格不超过当前行。
            短尺度价格无需提前产生反馈；Eq.(3) 的完整向量要等所有尺度到期。
            """
            nonlocal matured_orders
            while queue and times[queue[0]['row']] + longest <= times[now]:
                item = queue.popleft()
                origin = item['row']
                targets = times[origin] + hs
                indices = np.searchsorted(times[:now + 1], targets)
                safe = np.minimum(indices, now)
                if (indices > now).any() or not np.array_equal(times[safe], targets):
                    status = 'missing_target'
                elif (blocks[safe] != blocks[origin]).any():
                    status = 'crossed_gap_or_session'
                else:
                    status = 'matured'
                statuses[kind][status] += 1
                audit_version(item.get('model_version_id'), kind + '_' + status)
                if periodic and kind == 'OE':
                    # 单笔回调仅解决桶中一单；缺格也必须解决，不能从分母删除。
                    score = (self.reward.score(self.reward.features_from_prices(mid[origin], mid[safe], item['side']))
                             if status == 'matured' else None)
                    periods.resolve(item['period_key'], status, score)
                if status != 'matured':
                    continue
                values = self.reward.features_from_prices(mid[origin], mid[safe], item['side'])
                if kind == 'OE':
                    feature_sum[:] += values
                    matured_orders += 1
                if selector.reward_type == kind:
                    score = self.reward.score(values)
                    # 固定基线只记录反馈；传入从回放起点开始的毫秒，而不是行号。
                    if not periodic:
                        selector.observe(item['owner'], score, int((times[now] - times[0]) // 1_000_000))
                    if detail:
                        observations.append(dict(kind=kind, owner=item['owner'], reward=score,
                            origin_time=timestamp(times[origin]), due_time=timestamp(targets[-1]),
                            observed_time=timestamp(times[now]), target_times=[timestamp(t) for t in targets]))
                        if versions is not None:
                            observations[-1]['model_version_id'] = item['model_version_id']
                        if periodic:
                            observations[-1]['updates_selector'] = False

        for row, clock in enumerate(times):
            terminal = row == n - 1 and self.force_replay_end
            segment_end = ends[segments[row]]
            tradable = clock < segment_end
            scheduled_exit = tradable and clock + step >= segment_end
            reason = None
            if changed[row]:
                reason = ('session_transition' if sessions[row] != sessions[row - 1]
                          else 'scheduled_break' if segments[row] != segments[row - 1] else 'gap')
            if scheduled_exit:
                reason = 'scheduled_exit'
            if terminal and tradable:
                reason = 'replay_end'
            fills = []
            intent_version = None
            if versions is not None and row and versions[row] != versions[row - 1]:
                # 只根据当前可见换版撤销旧意图；仓位和成熟队列继续保留原身份。
                cancelled['model_version_change'] += len(intents)
                intents.clear()
            if reason or not tradable:
                cancelled[reason or 'closed_boundary'] += len(intents)
                intents.clear()
                if tradable and account.position:
                    fills = account.advance(int(clock), 0, account.owner, mid[row], bid[row], ask[row], terminal=True)
                    exits[reason] += 1
            else:
                eligible = []
                while intents and intents[0]['due'] <= clock:
                    eligible.append(intents.popleft())
                if eligible:
                    # 若到期意图合并到同一行情，只执行最新一个，显式记录其余被取代。
                    cancelled['superseded'] += len(eligible) - 1
                    intent = eligible[-1]
                    intent_version = intent.get('model_version_id')
                    fills = account.advance(int(clock), intent['side'], intent['owner'], mid[row], bid[row], ask[row])
                    for fill in fills:
                        fill['intent_time'] = timestamp(intent['origin'])
                        fill['execution_due_time'] = timestamp(intent['due'])
            for fill in fills:
                fill['execution_reason'] = reason or 'delayed_intent'
                order = dict(row=row, side=fill['side'], owner=fill['owner'])
                if versions is not None:
                    # 退出账本和 OE 沿用原开仓身份；触发退出的新信号身份另列。
                    fill['model_version_id'] = intent_version if fill['opening'] else position_version
                    fill['trigger_model_version_id'] = intent_version
                    audit_version(fill['model_version_id'], 'fills')
                    order['model_version_id'] = fill['model_version_id']
                    if fill['opening']:
                        position_version = fill['model_version_id']
                    else:
                        account.trades[-1].update(model_version_id=position_version,
                            exit_trigger_model_version_id=intent_version)
                        position_version = None
                if periodic:
                    # 按真实成交时刻归期间，退出的模型/版本仍归原仓位。
                    order['period_key'] = periods.add_order(sessions[row],
                        order.get('model_version_id', 'unversioned'), fill['owner'], int(clock))
                    fill['reward_period_index'] = order['period_key'][2]
                orders.append(order)
            mature(orders, row, 'OE')
            if not periodic:
                mature(signals, row, 'ME')
            else:
                # 顺序为实际成交 → 完整到期标签 → 已结束期间 → 本 tick 新选择。
                periods.advance(int(clock))
            gross = account.gross_equity_change(mid[row])
            gross_values.append(gross)
            net_values.append(gross - account.costs)
            if available[row] and tradable and not scheduled_exit and not terminal:
                elapsed_ms = int((clock - times[0]) // 1_000_000)
                owner = (periods.choose(sessions[row], versions[row] if versions is not None else 'unversioned', int(clock))
                         if periodic else selector.select_model(elapsed_ms))
                prediction = (0. if selector.flat else float(predictions[row].mean())
                              if selector.ensemble else predictions[row, owner])
                direction = int(prediction > self.threshold) - int(prediction < -self.threshold)
                signal = dict(row=row, side=direction, owner=owner)
                intent = dict(origin=int(clock), due=int(clock) + self.latency_ms * 1_000_000,
                              side=direction, owner=owner)
                if versions is not None:
                    signal['model_version_id'] = intent['model_version_id'] = versions[row]
                    audit_version(versions[row], 'decisions')
                if not periodic:
                    signals.append(signal)
                intents.append(intent)
                if detail:
                    decisions.append(dict(ts_event=timestamp(clock), owner=owner, direction=direction))
                    if versions is not None:
                        decisions[-1]['model_version_id'] = versions[row]

        equity = np.r_[self.initial_capital, self.initial_capital + np.asarray(net_values)]
        peaks = np.maximum.accumulate(equity)
        holding = np.array([t['holding_ms'] for t in account.trades])
        for kind, queue in [('OE', orders), ('ME', signals)]:
            statuses[kind]['unmatured_at_end'] = len(queue)
            for item in queue:
                audit_version(item.get('model_version_id'), kind + '_unmatured_at_end')
        points = sorted(set([0, n - 1] + list(range(0, n, max(1, n // 300)))))
        result = dict(strategy=selector.name, replay_rows=n, feature_valid_rows=int(available.sum()),
            decision_count=len(selector.action_history), total_trades=len(account.trades), total_fills=len(account.fills),
            gross_pnl_usd=float(gross_values[-1]), net_pnl_usd=float(net_values[-1]), friction_usd=float(account.costs),
            initial_capital=self.initial_capital, quantity=self.quantity,
            net_return=float(net_values[-1] / self.initial_capital),
            max_drawdown=float(((peaks - equity) / peaks).max()), max_drawdown_usd=float((peaks - equity).max()),
            terminal_position=account.position, terminal_position_liquidated=account.position == 0,
            holding_ms_mean=float(holding.mean()) if len(holding) else None,
            matured_order_count=matured_orders, order_feature_expectation_defined=bool(matured_orders),
            order_feature_expectation=(feature_sum / matured_orders if matured_orders else feature_sum).tolist(),
            reward_horizon_unit='milliseconds', reward_horizons_ms=self.reward.horizons,
            reward_definition=self.reward.definition, reward_weights=self.reward.weights.tolist(),
            reward_cost_mode='PaperOE-no-cost', reward_status={k: dict(v) for k, v in statuses.items()},
            observed_rewards=len(selector.reward_history), reward_type=selector.reward_type,
            unmatured_fill_rewards_at_end=len(orders), risk_exits=dict(exits),
            cancelled_intents=dict(cancelled), unexecuted_intents_at_end=len(intents),
            latency_ms=self.latency_ms, holding_review_ms=self.holding_review_ms, interval_ms=self.interval_ms,
            action_distribution=selector.get_selection_distribution(),
            timestamps=[timestamp(times[i]) for i in points],
            net_curve=[float(net_values[i] / self.initial_capital) for i in points],
            gross_curve=[float(gross_values[i] / self.initial_capital) for i in points],
            selector_protocol='static_baseline',
            force_replay_end=self.force_replay_end,
            execution_assumption='Aggressive one-contract quote simulation; no queue, impact or margin model.')
        if periodic:
            period_results = periods.finish(int(times[-1]))
            result.update(selector_protocol='fixed_period_OE_' + selector.mode,
                selection_period_ms=selector.period_ms, exploration_c_price=selector.c,
                period_feedback=period_results, period_selections=selector.period_selections,
                period_status_counts=dict(Counter(p['status'] for p in period_results)),
                selector_version_statistics=selector.summary(),
                period_assumptions='session_open_wall_clock_hold_choice_complete_orders_equal_period_W_version_separated',
                model_switch_policy='keep_position_original_owner_and_due_intents_until_execution_or_version_change')
            result['reward_status'].pop('ME')  # 本协议没有模型预测 ME 队列，不声称实现 ME 学习。
        if detail:
            result.update(fills=account.fills, trades=account.trades, decisions=decisions,
                          reward_observations=observations, equity_usd=equity[1:].tolist())
        if versions is not None:
            result.update(model_versions_seen=sorted(set(versions) - {None}),
                model_version_audit={k: dict(v) for k, v in version_audit.items()},
                terminal_position_model_version_id=position_version,
                model_version_policy='cancel_old_intents_keep_open_owner_and_pending_reward_identity')
        return result
