"""真实多周 OE 就绪审计：成熟校准覆盖、专家/动作一致性与因果周版本。

物理第3–4页 §3.2 要求不同历史/特征及每周更新，第4–5页 Algorithm 1、
Eq.(2)–(5) 需要能观察的多尺度奖励。这里仅回放校准静态参照；后段只
检查因果预测和质量，避免在尚不可执行的奖励上重复大量动态收益实验。
版本可用、奖励可学和完整 FMATO 复现是不同完成条件，报告分别说明。
"""
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.artifacts import pretty_json
from src.config import BASE_DIR
from src.learned_oe_experiment import execution_gate
from src.library_oe import experiment_code_hashes, freeze_library_oe, load_library, replay_candidates
from src.session_experiment import audit_horizons, fingerprint, read_day
from src.snapshot_backtest import PreparedDatasetReader
from src.snapshot_dataset import sha256_file, utc_ns
from src.snapshot_irl import learn_calibration_reward, pooled_policy_statistics
from src.sum_only_reward import learn_sum_only
from src.weekly_library import history_samples, predict_library


def code_hashes():
    """实际调用的校准/周库/等式学习/门控和新入口都绑定，不改旧源代码。"""
    return experiment_code_hashes() | {p: sha256_file(BASE_DIR/p) for p in
        ('src/learned_oe_experiment.py', 'src/sum_only_reward.py',
         'src/multiweek_readiness.py', 'run_multiweek_readiness.py')}


def freeze_readiness(config_path, library_path):
    """先固定连续阶段与跨周数量，不消费收益；沿用原年龄/执行/专家参数。

同一 ISO 周的中途启动不能算成另一个周。每个预期版本至少须有一个协议
内校准/后段 session；只有离线建出的未来版本不能满足真实跨周覆盖。
"""
    path = Path(config_path).resolve(); config = json.loads(path.read_text())
    keys = {'schema_version', 'purpose', 'oe_protocol_file', 'minimum_weekly_versions'}
    if (set(config)-keys-{'notes'} or keys-set(config) or config['schema_version'] != 1
            or config['purpose'] != 'development_multiweek_OE_readiness'
            or type(config['minimum_weekly_versions']) is not int or config['minimum_weekly_versions'] < 2):
        raise ValueError('Need explicit multiweek development readiness configuration')
    base = freeze_library_oe(path.parent/config['oe_protocol_file'], library_path)
    library, calendar = load_library(library_path)
    schedule = library['plan']['schedule']; minimum = config['minimum_weekly_versions']
    weeks = {tuple(pd.Timestamp(v['update_session']).isocalendar()[:2]) for v in schedule}
    if len(weeks) < minimum: raise ValueError('Not enough distinct weekly versions')
    protocol = base['cases'][0]['session_binding']['protocol']
    days = [d for phase in ('calibration','validation','test') for d in protocol['sessions'][phase]]
    used = {max(i for i,v in enumerate(schedule) if calendar.sessions[d]['open'] >= pd.Timestamp(v['available_at_utc']))
            for d in days}
    if used != set(range(len(schedule))): raise ValueError('Every scheduled version needs declared real quote sessions')
    plan = dict(schema_version=1, plan_kind='frozen_multiweek_OE_readiness', config=config,
        config_source=dict(path=str(path), sha256=sha256_file(path)), base_oe_plan=base,
        expected_update_sessions=[v['update_session'] for v in schedule], code_sha256=code_hashes(),
        stage_roles=dict(train='bootstrap_library_only', calibration='static_accounts_expert_reward_coverage',
                         validation='prediction_and_quality_audit_only', test='development_prediction_and_quality_audit_only'),
        holdout_claim='none_all_inspected_dates_are_development', age_selection='none',
        dynamic_backtest_policy='not_run_in_readiness_audit')
    plan['plan_sha256'] = fingerprint(plan)
    return plan


def calibration_prefix_coverage(daily, policies, horizons, minimum):
    """按已结束校准 session 的前缀显示成熟覆盖进展，不逐前缀择优学习。

各日 μ 仍按成熟订单数合并；没有成熟订单不是零期望。此函数只记资格
数量和原始覆盖，不改变最终专家、门槛、年龄或校准终点。
"""
    output = []; names = [p['policy_id'] for p in policies if p['source'] == 'weekly_library']
    for stop in range(1, len(daily)+1):
        rows = pooled_policy_statistics(daily[:stop], policies, horizons)
        selected = [r for r in rows if r['policy_id'] in names]
        missing = [r['policy_id'] for r in selected if not r['pnl_aggregation_defined']
                   or r['matured_order_count'] < minimum]
        output.append(dict(through_session=daily[stop-1]['session_id'], calibration_sessions=stop,
            library_policies=len(names), observed_library_policies=len(names)-len(missing),
            missing_library_policy_ids=missing,
            matured_orders_by_policy={r['policy_id']: r['matured_order_count'] for r in selected}))
    return output


def training_visibility_audit(reader, case, config):
    """重算每个实际历史日的样本时刻，核对记录数、purge与周版可用时间。

重算只用于独立审计，不回馈模型。训练窗口来自冻结规则；所有训练 session
必须在该版本可用前结束，预测标签必须已成熟，且每日尾部留出最长奖励
purge。即使后来成为验证日，也只允许在后续周版本中因果地使用它。
"""
    cache = {}; output = []
    for version in case['versions']:
        available = pd.Timestamp(version['available_at_utc']).value; models = []
        for model in version['models']:
            days = version['training_sessions'][str(model['history_session_count'])]
            if [r['session_id'] for r in model['training_sessions']] != days:
                raise ValueError('Recorded model history differs from scheduled history')
            counts = []; latest = None
            for day in days:
                close = reader.calendar.sessions[day]['close'].value
                if close >= available: raise ValueError('Training session is not closed before version availability')
                if day not in cache:
                    _, _, times = history_samples(reader, day, config); cache[day] = times
                times = cache[day]; last = int(times.max()) if len(times) else None
                if last is not None:
                    if last + config['purge_ms']*1000000 > close:
                        raise ValueError('Training origin violates longest-horizon daily purge')
                    if last + config['prediction_horizon_ms']*1000000 >= available:
                        raise ValueError('Training label is not mature before version availability')
                    latest = last if latest is None else max(latest, last)
                counts.append(dict(session_id=day, samples=len(times)))
            if counts != model['training_sessions'] or sum(r['samples'] for r in counts) != model['training_rows']:
                raise ValueError('Recomputed training coverage disagrees with model artifact')
            models.append(dict(model_id=model['model_id'], status=model['status'], training_rows=model['training_rows'],
                training_sessions=counts, latest_training_origin_ns=latest,
                latest_prediction_label_maturity_ns=latest+config['prediction_horizon_ms']*1000000 if latest is not None else None))
        output.append(dict(update_session=version['update_session'], version_sha256=version['version_sha256'],
            available_at_ns=available, training_visibility_verified=True, models=models))
    return output


def prediction_day_audit(reader, day, case, horizons):
    """先从当时可见特征预测，再统计全尺度质量；未来有效标记不决定预测资格。

每行版本独立与“当时最新已可用版本”核对，保存 N×K 预测字节哈希及所有
候选的可预测量。质量只作诊断；全尺度标签更多不代表旧报价可真实成交。
"""
    frame = read_day(reader, day); matrix, identities = predict_library(case['versions'], frame, reader.report['interval_ms'])
    available = np.array([pd.Timestamp(v['available_at_utc']).value for v in case['versions']], dtype=np.int64)
    positions = np.searchsorted(available, utc_ns(frame.ts_event), side='right')-1
    expected = np.array([case['versions'][i]['version_sha256'] if i >= 0 else None for i in positions], dtype=object)
    if not np.array_equal(identities, expected): raise ValueError('Prediction used a future or stale version')
    rows = []
    for version in case['versions']:
        keep = identities == version['version_sha256']
        if not keep.any(): continue
        finite = np.isfinite(matrix[keep])
        rows.append(dict(update_session=version['update_session'], version_sha256=version['version_sha256'],
            quote_rows=int(keep.sum()), all_models_predictable_rows=int(finite.all(axis=1).sum()),
            candidate_predictable_rows={m['model_id']: int(finite[:,i].sum()) for i,m in enumerate(version['models'])}))
    quality = audit_horizons(frame, reader.calendar, horizons)
    blocks = frame.groupby('continuous_block').ts_event.agg(['min','max'])
    quality['longest_continuous_ms'] = float(((blocks['max']-blocks['min']).dt.total_seconds()*1000).max())
    return dict(session_id=day, versions=rows, future_version_rows=0, horizon_audit=quality,
        prediction_matrix_sha256=hashlib.sha256(matrix.astype('<f8').tobytes()).hexdigest(),
        prediction_shape=list(matrix.shape), future_labels_used_for_prediction=False)


def run_readiness(plan_path, *, detail=False):
    """重算校准静态账本后学习，再检查后段版本；后段没有策略收益或参数选择。"""
    plan = json.loads(Path(plan_path).read_text()); base = plan.get('base_oe_plan', {})
    for value, kind, hashes in ((plan,'frozen_multiweek_OE_readiness',code_hashes()),
            (base,'frozen_library_time_OE_IRL',experiment_code_hashes())):
        if (value.get('schema_version') != 1 or value.get('plan_kind') != kind
                or value.get('plan_sha256') != fingerprint(value) or value.get('code_sha256') != hashes):
            raise ValueError('Frozen multiweek plan/source integrity check failed')
    source = base['library_source']
    if sha256_file(source['path']) != source['sha256']: raise ValueError('Library artifact changed')
    library, calendar = load_library(source['path']); cases = []
    if library['plan']['config'] != base['library_configuration']: raise ValueError('Library configuration changed')
    with threadpool_limits(limits=1):
        for frozen in base['cases']:
            binding = frozen['session_binding']; protocol = binding['protocol']
            if binding['plan_sha256'] != fingerprint(binding): raise ValueError('Session binding changed')
            reader = PreparedDatasetReader(binding['dataset']['path'],calendar=calendar,
                allow_partial=protocol['allow_partial'], include_degraded=protocol['include_degraded'])
            if reader.provenance != binding['dataset']: raise ValueError('Prepared dataset changed')
            case = next(c for c in library['age_cases'] if c['max_age_ms'] == frozen['max_age_ms'])
            training = training_visibility_audit(reader,case,base['library_configuration'])
            daily, audits = replay_candidates(reader,protocol['sessions']['calibration'],case,base['policies'],protocol,detail=detail)
            statistics = pooled_policy_statistics(daily,base['policies'],protocol['reward_horizons_ms'])
            fits = {name: learn_calibration_reward(statistics,protocol['reward_horizons_ms'],base['irl_config'],name)
                    for name in base['irl_config']['weight_constraints']}
            fits['sum_only'] = learn_sum_only(statistics,protocol['reward_horizons_ms'],base['irl_config'],fits.get('signed_box',fits['simplex']))
            gate = execution_gate(fits['sum_only'],base['policies'])
            identity = fingerprint(dict(statistics=statistics,reward_fits=fits,execution_gate=gate))
            phases = {}; covered = Counter()
            for phase in ('calibration','validation','test'):
                rows = [prediction_day_audit(reader,d,case,protocol['reward_horizons_ms']) for d in protocol['sessions'][phase]]
                for row in rows:
                    for version in row['versions']: covered[version['update_session']] += version['all_models_predictable_rows']
                phases[phase] = dict(prediction_audits=rows, strategy_profit_evaluated=phase=='calibration', calibration_sha256=identity)
                print(f"完成 {frozen['max_age_ms']}ms {phase} 因果版本/质量检查",flush=True)
            missing = [d for d in plan['expected_update_sessions'] if covered[d] == 0]
            readiness = dict(ready_for_development_online_run=gate['usable_for_execution'] and not missing,
                real_weekly_versions_verified=not missing, missing_predictable_update_sessions=missing,
                all_models_predictable_rows_by_version=dict(covered), dynamic_backtest_run=False,
                formal_replication_ready=False)
            cases.append(dict(max_age_ms=frozen['max_age_ms'],training_visibility_audits=training,
                calibration=dict(daily=daily,statistics=statistics,horizon_audits=audits,
                    prefix_coverage=calibration_prefix_coverage(daily,base['policies'],protocol['reward_horizons_ms'],base['irl_config']['minimum_matured_orders'])),
                reward_fits=fits, execution_gate=gate, calibration_sha256=identity, phases=phases,
                readiness=readiness, active_reward_weights=None))
            print(f"{frozen['max_age_ms']}ms 校准 {fits['sum_only']['status']}；门控 {gate['reasons']}；跨周覆盖 {not missing}",flush=True)
    if plan['code_sha256'] != code_hashes() or sha256_file(source['path']) != source['sha256']:
        raise ValueError('Source/library changed during readiness audit')
    return dict(schema_version=1,result_kind='multiweek_OE_readiness_development',plan=plan,
        plan_sha256=plan['plan_sha256'],age_cases=cases,selected_age_ms=None,
        environment=dict(python=platform.python_version(),**{n:importlib.metadata.version(n) for n in
            ('numpy','pandas','pyarrow','scikit-learn','scipy','threadpoolctl')}),
        limits=['仅校准静态账本与预测/质量就绪审计，不是跨周动态交易或正式全量实验。',
                '成熟订单子集、净利专家、现金零向量与动作门控均为既有工程口径。',
                '1/2个session历史、盘口深度特征、树抽样和训练瞬时可用仍为工程近似。',
                '按周可使用已结束的较早开发日，不使用尚不可见标签；奖励不跨周重学。',
                '所有年龄保留；零成熟/现金专家/不可表示/模型无预测仍保留阻断。',
                '报价年龄放宽可能增加标签但不证明可成交；没有新增未触碰正式测试集声明。',
                '仍缺时间ME学习、四变体整体对照、动态多周交易与正式长期方案。'])


def render_readiness(result):
    """同源报告全部年龄的就绪、前缀覆盖、版本与校准账本，不展示后段收益。"""
    lines = ['# 真实多周 OE 就绪审计','',f"计划 `{result['plan_sha256']}`；仅开发用途。",'',
        '| 年龄 ms | 校准拟合 | 专家 | 可执行奖励 | 真实周版本 | 在线开发就绪 | 原因 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for c in result['age_cases']:
        g=c['execution_gate'];r=c['readiness'];f=c['reward_fits']['sum_only']
        lines.append(f"| {c['max_age_ms']} | {f['status']} | {g['expert_policy_id']} | {g['usable_for_execution']} | "
            f"{r['real_weekly_versions_verified']} | {r['ready_for_development_online_run']} | {g['reasons']} |")
    lines += ['','## 校准前缀成熟覆盖','', '| 年龄 ms | 截止 session | 已观察库候选 | 缺观察候选 |',
        '| --- | --- | --- | --- |']
    for c in result['age_cases']:
        for row in c['calibration']['prefix_coverage']:
            lines.append(f"| {c['max_age_ms']} | {row['through_session']} | {row['observed_library_policies']}/{row['library_policies']} | {row['missing_library_policy_ids']} |")
    lines += ['','## 校准全部静态账本','', '| 年龄 ms | 候选 | 成交 | 成熟订单 | 净利 USD | 成熟成交比例 |',
        '| --- | --- | --- | --- | --- | --- |']
    for c in result['age_cases']:
        for row in c['calibration']['statistics']:
            lines.append(f"| {c['max_age_ms']} | {row['policy_id']} | {row['total_fills']} | {row['matured_order_count']} | {row['net_pnl_usd']} | {row['matured_fraction_of_fills']} |")
    lines += ['','## 因果周版本与长期标签','', '| 年龄 ms | 阶段 | session | 使用版本 | 全尺度有效特征行 | 最长连续 ms |',
        '| --- | --- | --- | --- | --- | --- |']
    for c in result['age_cases']:
        for phase,data in c['phases'].items():
            for row in data['prediction_audits']:
                q=row['horizon_audit']
                lines.append(f"| {c['max_age_ms']} | {phase} | {row['session_id']} | {[v['update_session'] for v in row['versions']]} | {q['all_horizons_feature_valid_rows']} | {q['longest_continuous_ms']} |")
    lines += ['','## 学习与就绪诊断','']
    for c in result['age_cases']:
        lines += [f"### {c['max_age_ms']}ms",'','```json',pretty_json(dict(reward_fits=c['reward_fits'],
            execution_gate=c['execution_gate'],readiness=c['readiness'])),'```','']
    lines += ['## 限制','']+['- '+s for s in result['limits']]
    return '\n'.join(lines)+'\n'
