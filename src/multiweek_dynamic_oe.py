"""真实多周 OE 动态对照：两种工程专家来源、同一完整候选矩阵与执行条件。

对应论文物理第4页 Algorithm 1 和第5页 Algorithms 2–3。复用既有求解器、
交易器和选择器，仅连接 PR #18 的专家来源对照与真实多周行情。每个年龄
先重新回放全部校准日，冻结两组奖励，再评价后段；不消费旧归档中的权重。
每日账户/反馈重启，周模型因果更新，奖励不重学，不能称为连续实盘资金链。
"""
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.ars_experiment import code_hashes as ars_hashes, freeze_ars_experiment, replay_ars_day
from src.config import BASE_DIR, INSTRUMENT_CONFIG
from src.expert_scope import SCOPES, fit_scope
from src.learned_oe_experiment import (LEARNED_STRATEGIES, blocked_result,
    code_hashes as learned_hashes, replay_learned_day)
from src.library_oe import experiment_code_hashes as oe_hashes, freeze_library_oe, load_library, replay_candidates
from src.multiweek_readiness import prediction_day_audit, training_visibility_audit
from src.period_experiment import code_hashes as period_hashes
from src.session_experiment import fingerprint, publish_bundle, summarize_days
from src.snapshot_backtest import PreparedDatasetReader
from src.snapshot_dataset import sha256_file
from src.snapshot_irl import pooled_policy_statistics


def code_hashes():
    """绑定全部实际执行依赖；不修改旧入口或旧模型库的代码身份。"""
    return learned_hashes() | {p: sha256_file(BASE_DIR / p) for p in
        ('src/expert_scope.py', 'src/multiweek_readiness.py',
         'src/multiweek_dynamic_oe.py', 'run_multiweek_dynamic_oe.py')}


def environment_versions():
    """恢复时也检查依赖版本，不能把两套数值环境产生的日账本混成一次运行。"""
    return dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in
        ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'scipy', 'threadpoolctl')})


class ReplayCheckpoints:
    """按完整日原子保存可重算结果，解决长回放中断后内存账本丢失的问题。

    缓存身份包括完整冻结计划（因而包括代码/数据/模型）、明细开关与实际
    依赖环境。每份日结果另有内容哈希；错误计划、损坏内容直接拒绝，不当成
    缓存缺失重跑来掩盖问题。新目录原子发布，不覆盖既有日结果。不开启时
    只在内存运行，最终JSON不记录缓存命中情况，便于逐字段比较恢复一致性。
    """
    def __init__(self, directory, plan_sha256, detail):
        self.directory = Path(directory).resolve() if directory is not None else None
        self.binding = dict(schema_version=1, plan_sha256=plan_sha256,
                            detail=detail, environment=environment_versions())
        if self.directory is not None:
            if self.directory.exists():
                header = self.directory / 'binding.json'
                if not header.is_file() or json.loads(header.read_text()) != self.binding:
                    raise ValueError('Checkpoint plan/detail/environment binding differs')
            else:
                publish_bundle(self.binding, '# 多周日回放检查点\n\n不是最终实验结果。\n',
                               self.directory, 'binding.json')

    def load(self, key):
        """只有完整发布且身份一致的单位可恢复；未发布的临时目录不算完成。"""
        if self.directory is None or not (self.directory / key).exists():
            return None
        path = self.directory / key / 'checkpoint.json'
        if not path.is_file():
            raise ValueError('Incomplete checkpoint directory')
        row = json.loads(path.read_text())
        if (row.get('binding') != self.binding or row.get('key') != key
                or row.get('payload_sha256') != fingerprint(dict(payload=row.get('payload')))):
            raise ValueError('Checkpoint identity or payload integrity differs')
        return row['payload']

    def save(self, key, payload):
        """每个单位只保存一次；中断发生在写入中时不会出现可误用的半份账本。"""
        if self.directory is not None:
            row = dict(binding=self.binding, key=key, payload=payload,
                       payload_sha256=fingerprint(dict(payload=payload)))
            publish_bundle(row, '# 完整日回放检查点\n\n须配合冻结计划校验后使用。\n',
                           self.directory / key, 'checkpoint.json')


def strategy_names(scope):
    """全候选沿用旧名称；库内组单独命名，避免覆盖旧规则的阻断或交易结果。"""
    if scope not in SCOPES:
        raise ValueError('Unknown expert scope')
    return list(LEARNED_STRATEGIES) if scope == 'all_candidates' else [
        name.replace('Learned-OE-', 'LibraryExpert-OE-') for name in LEARNED_STRATEGIES]


def validate_config(config):
    """两个来源必须同时保留；只激活预声明的 sum_only，不按收益选择约束。"""
    keys = {'schema_version', 'purpose', 'ars_protocol_file', 'expert_scopes',
            'active_constraint', 'minimum_evaluation_weekly_versions'}
    if (set(config) - keys - {'notes'} or keys - set(config) or config['schema_version'] != 1
            or config['purpose'] != 'development_multiweek_dynamic_OE'
            or config['expert_scopes'] != list(SCOPES) or config['active_constraint'] != 'sum_only'
            or type(config['minimum_evaluation_weekly_versions']) is not int
            or config['minimum_evaluation_weekly_versions'] < 2):
        raise ValueError('Need both expert scopes, sum_only and multiple evaluation weeks')


def evaluation_schedule(library, calendar, protocol, minimum):
    """只有验证/开发测试实际包含的周才计数，不能拿校准周凑动态跨周覆盖。"""
    schedule = library['plan']['schedule']
    days = protocol['sessions']['validation'] + protocol['sessions']['test']
    expected = {}
    for day in days:
        visible = [v for v in schedule if pd.Timestamp(v['available_at_utc']) <= calendar.sessions[day]['open']]
        if not visible:
            raise ValueError('Evaluation session has no available model version')
        expected[day] = visible[-1]['update_session']
    weeks = {tuple(pd.Timestamp(d).isocalendar()[:2]) for d in expected.values()}
    if len(weeks) < minimum:
        raise ValueError('Not enough distinct evaluation weekly versions')
    return expected


def freeze_dynamic(config_path, library_path):
    """评价前绑定原18个控制、两组学习策略、全部数据和周计划，不计算收益。"""
    path = Path(config_path).resolve()
    config = json.loads(path.read_text()); validate_config(config)
    ars_path = path.parent / config['ars_protocol_file']
    ars = freeze_ars_experiment(ars_path, library_path)
    period_path = ars_path.parent / ars['config']['period_protocol_file']
    base = ars['base_period_plan']
    oe = freeze_library_oe(period_path.parent / base['config']['oe_protocol_file'], library_path)
    if oe['cases'] != base['cases'] or oe['library_source'] != base['library_source']:
        raise ValueError('Calibration and execution bindings differ')
    library, calendar = load_library(library_path)
    expected = evaluation_schedule(library, calendar, base['cases'][0]['session_binding']['protocol'],
                                   config['minimum_evaluation_weekly_versions'])
    plan = dict(schema_version=1, plan_kind='frozen_multiweek_dynamic_OE', config=config,
        config_source=dict(path=str(path), sha256=sha256_file(path)), base_ars_plan=ars, base_oe_plan=oe,
        strategies=ars['strategies'] + strategy_names('all_candidates') + strategy_names('online_library'),
        expected_evaluation_versions=expected, code_sha256=code_hashes(),
        stage_roles=dict(calibration='replay_static_accounts_then_freeze_both_rewards',
                         validation='dynamic_development_no_retuning', test='dynamic_development_no_retuning'),
        holdout_claim='none_all_dates_already_development', selected_age_ms=None, selected_expert_scope=None,
        reward_update_policy='once_after_calibration_never_during_evaluation',
        account_policy='independent_daily_accounts_and_feedback',
        activation_policy='sum_only_with_original_expert_and_full_library_gate')
    plan['plan_sha256'] = fingerprint(plan)
    return plan


def fit_groups(statistics, horizons, config, policies):
    """P×H 成熟订单均值只来自校准；两组仅改专家来源，完整 oracle 保持一致。"""
    groups = {scope: fit_scope(statistics, horizons, config, policies, scope) for scope in SCOPES}
    for constraint in ('simplex', 'signed_box', 'sum_only'):
        if (groups['all_candidates']['reward_fits'][constraint]['eligible_policy_ids'] !=
                groups['online_library']['reward_fits'][constraint]['eligible_policy_ids']):
            raise ValueError('Expert scope changed the reward comparison matrix')
    for group in groups.values():
        # fit_scope 的默认输出仅供诊断；本入口明确记录通过门控后将使用的权重。
        group['active_reward_weights'] = (group['reward_fits']['sum_only']['weights']
            if group['execution_gates']['sum_only']['usable_for_execution'] else None)
    return groups


def replay_scope(reader, day, case, protocol, plan, group, *, detail=False):
    """用原门控决定执行或显式阻断；名称变换只标识实验组，不改变选择过程。"""
    scope = group['expert_scope']; gate = group['execution_gates']['sum_only']
    names = strategy_names(scope)
    if not gate['usable_for_execution']:
        return [blocked_result(name, gate) | dict(expert_scope=scope) for name in names]
    weights = group['reward_fits']['sum_only']['weights']
    rows = replay_learned_day(reader, day, case, protocol, plan, weights, detail=detail)
    for name, row in zip(names, rows):
        row.update(strategy=name, expert_scope=scope)
    return rows


def audit_dynamic_result(result, versions, longest_ms, weights=None):
    """独立核对实际选择版本、全期间订单守恒与成熟时序；失败不以收益掩盖。

    输入是一日单策略结果。所有时间均为 UTC 纳秒；价格点奖励和美元账本
    分开。旧仓位反馈允许属于旧版，但决策必须使用当时最新已可用版本。
    ARS 可评分历史窗还须满足观察不越过当下、全部订单成熟及冻结权重内积。
    返回选择/反馈/历史评分数量；无选择不伪造跨周动态覆盖。
    """
    if result.get('run_status') == 'blocked':
        if result['net_pnl_usd'] is not None or result['total_fills'] is not None:
            raise ValueError('Blocked strategy must not have observed profit or fills')
        return dict(run_status='blocked', selections=0, mature_periods=0, history_windows=0, scored_windows=0,
                    selected_version_ids=[])
    feedback = result.get('period_feedback', [])
    selections = result.get('period_selections', [])
    if 'period_feedback' not in result:
        return dict(run_status='static_control')
    if sum(p['orders'] for p in feedback) != result['total_fills']:
        raise ValueError('Period ledger does not conserve fills')
    if sum(p['statuses'].get('matured', 0) for p in feedback) != result['matured_order_count']:
        raise ValueError('Period ledger does not conserve mature orders')
    if weights is not None and result['reward_weights'] != list(weights):
        raise ValueError('Execution changed frozen reward weights')
    by_id = {v['version_sha256']: v for v in versions}
    longest = longest_ms * 1_000_000
    mature = 0; histories = 0; scored = 0
    for p in feedback:
        if p['orders'] != p['pending'] + sum(p['statuses'].values()):
            raise ValueError('Unresolved orders missing from period denominator')
        if p['updated_selector']:
            if (p['status'] != 'matured' or not p['orders'] or p['pending']
                    or p['statuses'].get('matured', 0) != p['orders']
                    or p['observed_at_ns'] < max(p['end_ns'], p['last_origin_ns'] + longest)
                    or not np.isclose(p['reward'], p['reward_sum'] / p['orders'], rtol=1e-10, atol=1e-10)):
                raise ValueError('Invalid complete-period reward or maturity')
            mature += 1
    for p in selections:
        visible = [v for v in versions if pd.Timestamp(v['available_at_utc']).value <= p['selected_at_ns']]
        if not visible or visible[-1]['version_sha256'] != p['model_version_id']:
            raise ValueError('Dynamic selection used a future or stale version')
        for e in p.get('evaluations', []):
            if e['model_version_id'] != p['model_version_id']:
                raise ValueError('ARS score belongs to another model version')
            if e['source'] == 'live_selected_model':
                if e['live_period'] and e['live_period']['observed_at_ns'] > p['selected_at_ns']:
                    raise ValueError('ARS live feedback comes from the future')
                continue
            histories += 1
            if e['known_through_ns'] != p['selected_at_ns'] or e['window_end_ns'] > e['known_through_ns']:
                raise ValueError('Historical observation exceeds current time')
            if e['score'] is not None:
                version_start = pd.Timestamp(by_id[e['model_version_id']]['available_at_utc']).value
                if (e['status'] != 'matured' or not e['total_fills']
                        or e['matured_order_count'] != e['total_fills']
                        or e['reward_observation_latest_required_ns'] > e['known_through_ns']
                        or e['window_start_ns'] < version_start
                        or not np.isclose(e['score'], np.dot(result['reward_weights'], e['order_feature_expectation']),
                                          rtol=1e-10, atol=1e-10)):
                    raise ValueError('Historical reward is not fully mature or uses other weights')
                scored += 1
    stats = result['selector_version_statistics']
    if (sum(sum(v['visits']) for v in stats.values()) != len(selections)
            or sum(sum(v['feedback_counts']) for v in stats.values()) != mature):
        raise ValueError('Selector counts disagree with decisions or feedback')
    return dict(run_status='executed', selections=len(selections), mature_periods=mature,
        history_windows=histories, scored_windows=scored,
        selected_version_ids=sorted({p['model_version_id'] for p in selections}))


def summarize_phase(daily, strategies):
    """日账户仅作可解释加总；阻断是 None，无交易现金为0，反馈覆盖另列。"""
    output = []
    for name in strategies:
        rows = [next(r for r in d['results'] if r['strategy'] == name) for d in daily]
        blocked = [r.get('run_status') == 'blocked' for r in rows]
        if any(blocked):
            if not all(blocked):
                raise ValueError('A frozen calibration gate changed during evaluation')
            output.append(rows[0].copy()); continue
        total = summarize_days(daily, name) | dict(run_status='executed')
        for key in ('period_status_counts', 'history_status_counts'):
            counts = Counter()
            for row in rows:
                counts.update(row.get(key, {}))
            total[key] = dict(counts)
        for key in ('selections', 'mature_periods', 'history_windows', 'scored_windows'):
            total[key] = sum(d['dynamic_audits'][name].get(key, 0) for d in daily)
        total['cold_start_periods'] = sum(r.get('cold_start_periods', 0) for r in rows)
        output.append(total)
    return output


def run_dynamic(plan_path, *, detail=False, checkpoint_dir=None):
    """按年龄流式加载 session；两组校准一次冻结，18控制与6学习状态全保留。"""
    plan = json.loads(Path(plan_path).read_text()); validate_config(plan['config'])
    ars = plan['base_ars_plan']; base = ars['base_period_plan']; oe = plan['base_oe_plan']
    for value, kind, hashes in ((plan, 'frozen_multiweek_dynamic_OE', code_hashes()),
            (ars, 'frozen_history_OE_ARS', ars_hashes()), (base, 'frozen_period_OE_UCB', period_hashes()),
            (oe, 'frozen_library_time_OE_IRL', oe_hashes())):
        if (value.get('schema_version') != 1 or value.get('plan_kind') != kind
                or value.get('plan_sha256') != fingerprint(value) or value.get('code_sha256') != hashes):
            raise ValueError('Frozen dynamic plan/source integrity check failed')
    source = base['library_source']
    if sha256_file(source['path']) != source['sha256']:
        raise ValueError('Library artifact changed')
    library, calendar = load_library(source['path']); cases = []
    if library['plan']['config'] != base['library_configuration']:
        raise ValueError('Library configuration changed')
    if oe['cases'] != base['cases'] or oe['library_source'] != source:
        raise ValueError('Calibration and execution bindings differ')
    if plan['strategies'] != ars['strategies'] + strategy_names('all_candidates') + strategy_names('online_library'):
        raise ValueError('All control and learned strategies must be retained')
    expected = evaluation_schedule(library, calendar, base['cases'][0]['session_binding']['protocol'],
                                   plan['config']['minimum_evaluation_weekly_versions'])
    if plan['expected_evaluation_versions'] != expected:
        raise ValueError('Frozen evaluation schedule changed')
    checkpoints = ReplayCheckpoints(checkpoint_dir, plan['plan_sha256'], detail)
    with threadpool_limits(limits=1):
        for frozen in base['cases']:
            binding = frozen['session_binding']; protocol = binding['protocol']
            if binding['plan_sha256'] != fingerprint(binding):
                raise ValueError('Session binding changed')
            reader = PreparedDatasetReader(binding['dataset']['path'], calendar=calendar,
                allow_partial=protocol['allow_partial'], include_degraded=protocol['include_degraded'])
            if reader.provenance != binding['dataset']:
                raise ValueError('Prepared dataset changed')
            case = next(c for c in library['age_cases'] if c['max_age_ms'] == frozen['max_age_ms'])
            training = training_visibility_audit(reader, case, base['library_configuration'])
            daily, quality = [], []
            for day in protocol['sessions']['calibration']:
                key = f"age-{frozen['max_age_ms']}-calibration-{day}"
                cached = checkpoints.load(key)
                if cached is None:
                    rows, audits = replay_candidates(reader, [day], case, oe['policies'], protocol, detail=detail)
                    cached = dict(daily=rows, horizon_audits=audits)
                    checkpoints.save(key, cached)
                if [d['session_id'] for d in cached['daily']] != [day]:
                    raise ValueError('Checkpoint calibration day differs')
                daily += cached['daily']; quality += cached['horizon_audits']
                print(f"完成/恢复 {frozen['max_age_ms']}ms calibration {day}", flush=True)
            statistics = pooled_policy_statistics(daily, oe['policies'], protocol['reward_horizons_ms'])
            groups = fit_groups(statistics, protocol['reward_horizons_ms'], oe['irl_config'], oe['policies'])
            calibration = dict(daily=daily, statistics=statistics, horizon_audits=quality, scopes=groups)
            identity = fingerprint(dict(statistics=statistics, scopes=groups))
            print(f"{frozen['max_age_ms']}ms 两组校准冻结：" + str({s: g['execution_gates']['sum_only']
                  ['usable_for_execution'] for s, g in groups.items()}), flush=True)
            phases = {}; coverage = {name: set() for s in SCOPES for name in strategy_names(s)}
            for phase in ('validation', 'test'):
                daily = []
                for day in protocol['sessions'][phase]:
                    prediction = prediction_day_audit(reader, day, case, protocol['reward_horizons_ms'])
                    key = f"age-{frozen['max_age_ms']}-{phase}-{day}"
                    row = checkpoints.load(key); restored = row is not None
                    if restored:
                        if (row['session_id'] != day or row['calibration_sha256'] != identity
                                or row['prediction_audit'] != prediction):
                            raise ValueError('Checkpoint day/calibration/prediction differs')
                    else:
                        row = replay_ars_day(reader, day, case, protocol, ars, detail=detail)
                        for group in groups.values():
                            row['results'] += replay_scope(reader, day, case, protocol, plan, group, detail=detail)
                    if [r['strategy'] for r in row['results']] != plan['strategies']:
                        raise ValueError('Daily checkpoint must retain all strategies in order')
                    audits = {}
                    for r in row['results']:
                        weights = groups[r['expert_scope']]['reward_fits']['sum_only']['weights'] if 'expert_scope' in r else None
                        audits[r['strategy']] = audit_dynamic_result(r, case['versions'], max(protocol['reward_horizons_ms']), weights)
                        if r['strategy'] in coverage:
                            coverage[r['strategy']].update(audits[r['strategy']]['selected_version_ids'])
                    if restored and row['dynamic_audits'] != audits:
                        raise ValueError('Restored dynamic audit differs')
                    row.update(dynamic_audits=audits, prediction_audit=prediction, calibration_sha256=identity)
                    if not restored:
                        checkpoints.save(key, row)
                    daily.append(row)
                    print(f"{'恢复' if restored else '完成'} {frozen['max_age_ms']}ms {phase} {day}：24项策略状态及因果审计", flush=True)
                phases[phase] = dict(daily=daily, statistics=summarize_phase(daily, plan['strategies']),
                                    calibration_sha256=identity)
            expected_ids = {v['version_sha256'] for v in case['versions']
                            if v['update_session'] in set(plan['expected_evaluation_versions'].values())}
            cases.append(dict(max_age_ms=frozen['max_age_ms'], calibration=calibration,
                calibration_sha256=identity, training_visibility_audits=training, phases=phases,
                dynamic_version_coverage={name: dict(selected_version_ids=sorted(ids),
                    expected_version_ids=sorted(expected_ids), all_evaluation_versions_selected=ids == expected_ids)
                    for name, ids in coverage.items()}))
    if code_hashes() != plan['code_sha256'] or sha256_file(source['path']) != source['sha256']:
        raise ValueError('Source/library changed during dynamic experiment')
    return dict(schema_version=1, result_kind='multiweek_dynamic_OE_development', plan=plan,
        plan_sha256=plan['plan_sha256'], age_cases=cases, selected_age_ms=None, selected_expert_scope=None,
        formal_replication_ready=False, instrument=INSTRUMENT_CONFIG['CME_ES'],
        environment=environment_versions(),
        limits=['两种专家来源、净利专家、有限策略最大间隔及最优面最小L1均为明确工程假设。',
            '校准使用成熟订单子集；在线期间和历史窗口要求全部订单成熟，覆盖不能混称。',
            '只激活sum_only且必须通过原门控；负净利专家、无观察、零交易和求解失败全部保留。',
            '每个session账户/选择器/队列重启；模型因果周更新，奖励不重学，不是跨日连续资金链。',
            '全部年龄保留，C与执行成本沿用，长奖励造成的反馈稀疏和ARS窗口歧义仍存在。',
            '所有日期已有开发用途；CME替代数据和短训练历史不能复现论文原始数值。',
            '仍缺真实时间ME、完整四变体/基线消融和正式长期测试；不以盈利认定复现成功。'])


def render_dynamic(result):
    """同一JSON生成全部门控、利润、反馈覆盖和周版本结果，阻断保持None。"""
    lines = ['# 真实多周 OE 动态对照', '', f"冻结计划 `{result['plan_sha256']}`；全部日期仅开发用途。", '',
        '| 年龄 ms | 专家来源 | 专家 | 校准净利 USD | 拟合状态 | 可执行 | 阻断原因 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for c in result['age_cases']:
        for scope, g in c['calibration']['scopes'].items():
            f = g['reward_fits']['sum_only']; gate = g['execution_gates']['sum_only']; e = f['expert'] or {}
            lines.append(f"| {c['max_age_ms']} | {scope} | {e.get('selected_policy_id')} | {e.get('best_score')} | "
                         f"{f['status']} | {gate['usable_for_execution']} | {gate['reasons']} |")
    lines += ['', '## 全部策略与反馈覆盖', '',
        '| 年龄 ms | 阶段 | 策略 | 状态 | 成交 | 毛利 USD | 摩擦 USD | 净利 USD | 成熟期间/选择次数 | 可评分/全部历史窗 | 冷启动期间 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in result['age_cases']:
        for phase, data in c['phases'].items():
            for r in data['statistics']:
                lines.append(f"| {c['max_age_ms']} | {phase} | {r['strategy']} | {r['run_status']} | {r['total_fills']} | "
                    f"{r['gross_pnl_usd']} | {r['friction_usd']} | {r['net_pnl_usd']} | "
                    f"{r.get('mature_periods')}/{r.get('selections')} | {r.get('scored_windows')}/{r.get('history_windows')} | {r.get('cold_start_periods')} |")
    lines += ['', '成熟期间桶与选择次数不是同一总体，二者之比不能解释为概率。', '', '## 实际动态周版本', '',
              '| 年龄 ms | 策略 | 实际选择版本数 | 覆盖全部评价版本 |', '| --- | --- | --- | --- |']
    for c in result['age_cases']:
        for name, coverage in c['dynamic_version_coverage'].items():
            lines.append(f"| {c['max_age_ms']} | {name} | {len(coverage['selected_version_ids'])} | {coverage['all_evaluation_versions_selected']} |")
    lines += ['', '## 限制', ''] + ['- ' + s for s in result['limits']]
    return '\n'.join(lines) + '\n'
