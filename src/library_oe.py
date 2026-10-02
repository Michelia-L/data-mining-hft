"""把 §3.2 的轻模型版本接入真实成交 OE 与有限策略 Algorithm 1 近似。

候选身份/奖励权重在校准期冻结，数值模型按预声明周规则使用已结束历史更新。
这是可见性约束的开发 walk-forward，不是 Algorithm 2/3，也不是未触碰测试。
"""
import importlib.metadata
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from src.artifacts import pretty_json
from src.config import BASE_DIR, INSTRUMENT_CONFIG
from src.irl_reward import IRLRewardLearner
from src.model_selector import FlatSelector, SingleModelSelector
from src.session_experiment import audit_horizons, fingerprint, freeze_protocol, read_day
from src.snapshot_backtest import FEATURE_COLUMNS, PreparedDatasetReader
from src.snapshot_dataset import SessionCalendar, sha256_file
from src.snapshot_irl import (evaluate_frozen_weights, learn_calibration_reward, policy_specs,
    pooled_policy_statistics, validate_irl_config)
from src.time_execution import TimeExecutionEngine
from src.timed_snapshots import SNAPSHOT_SCHEMA
from src.weekly_library import (library_code_hashes, predict_library, predict_model,
    validate_library_config, weekly_schedule)


def experiment_code_hashes():
    """同时绑定建库、IRL、执行器和新入口；旧实验重跑须另存身份。"""
    return library_code_hashes() | {name: sha256_file(BASE_DIR / name) for name in
                                  ('src/snapshot_irl.py', 'src/library_oe.py', 'run_library_oe.py')}


def load_library(path):
    """仅接受当前代码生成、版本哈希完整的模型产物；不从诊断误差择优。"""
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    if artifact.get('schema_version') != 1 or artifact.get('result_kind') != 'weekly_light_library_development':
        raise ValueError('Unsupported library artifact')
    plan = artifact['plan']
    if artifact['plan_sha256'] != plan['plan_sha256'] or plan['plan_sha256'] != fingerprint(plan):
        raise ValueError('Library plan integrity check failed')
    if plan['code_sha256'] != library_code_hashes():
        raise ValueError('Library source changed; rebuild with a new frozen plan')
    calendar = SessionCalendar(plan['calendar']['path'])
    if calendar.sha256 != plan['calendar']['sha256']:
        raise ValueError('Library calendar changed')
    validate_library_config(plan['config'], calendar)
    if plan['schedule'] != weekly_schedule(plan['config'], calendar):
        raise ValueError('Library schedule changed')
    if [c['max_age_ms'] for c in artifact['age_cases']] != plan['config']['age_candidates_ms']:
        raise ValueError('Library artifact must retain every declared age')
    identities = None
    for case in artifact['age_cases']:
        if len(case['versions']) != len(plan['schedule']):
            raise ValueError('Need every scheduled library version')
        for version, scheduled in zip(case['versions'], plan['schedule']):
            if (version['version_sha256'] != fingerprint({k: v for k, v in version.items() if k != 'version_sha256'})
                    or any(version[k] != v for k, v in scheduled.items())):
                raise ValueError('Model version integrity or schedule check failed')
            ids = [m['model_id'] for m in version['models']]
            if not ids or len(ids) != len(set(ids)) or identities is not None and ids != identities:
                raise ValueError('Candidate identities must stay stable across ages/versions')
            identities = ids
    return artifact, calendar


def freeze_library_oe(config_path, library_path):
    """冻结模型文件、四段协议、有限候选与权约束，尚不计算交易收益。

    library 文件须由当前代码的 freeze/build 生成；若执行代码改变导致旧绑定
    不一致，保留旧产物并另存重建，不能把旧实验身份套到新代码。
    """
    config_path, library_path = Path(config_path).resolve(), Path(library_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    required = {'schema_version', 'purpose', 'session_protocol_file', 'irl_config_file',
                'model_update_policy', 'session_boundary_policy'}
    if (set(config) - (required | {'notes'}) or required - set(config) or config['schema_version'] != 1
            or config['purpose'] != 'development_library_time_OE_IRL'
            or config['model_update_policy'] != 'predeclared_weekly_visible'
            or config['session_boundary_policy'] != 'independent_daily_accounts_no_feedback_carry'):
        raise ValueError('Unsupported library OE configuration')
    artifact, calendar = load_library(library_path)
    irl_path = (config_path.parent / config['irl_config_file']).resolve()
    irl = json.loads(irl_path.read_text(encoding='utf-8'))
    validate_irl_config(irl)
    protocol_path = (config_path.parent / config['session_protocol_file']).resolve()
    cases = []
    for row in artifact['plan']['cases']:
        binding = freeze_protocol(protocol_path, row['dataset']['path'], calendar_path=calendar.path)
        protocol, library_config = binding['protocol'], artifact['plan']['config']
        if (binding['dataset'] != row['dataset'] or irl['age_candidates_ms'] != library_config['age_candidates_ms']
                or protocol['prediction_horizon_ms'] != library_config['prediction_horizon_ms']
                or max(protocol['reward_horizons_ms']) != library_config['purge_ms']
                or protocol['allow_partial'] != library_config['allow_partial']
                or protocol['include_degraded'] != library_config['include_degraded']
                or library_config['bootstrap_session'] != protocol['sessions']['calibration'][0]
                or artifact['plan']['schedule'][0]['training_sessions'][str(max(library_config['history_session_counts']))]
                   != protocol['sessions']['train']):
            raise ValueError('Library and execution training/quality/time protocols must agree')
        if any(d not in library_config['evaluation_sessions'] for phase in ('calibration', 'validation', 'test')
               for d in protocol['sessions'][phase]):
            raise ValueError('Execution sessions must be declared in library evaluation schedule')
        cases.append(dict(max_age_ms=row['max_age_ms'], session_binding=binding))
    specs = [s | dict(source='fixed_reference', model_id=None) for s in policy_specs(irl, protocol['threshold'])]
    specs += [dict(policy_id='Library-' + m['model_id'], model_id=m['model_id'], cash=False,
                   threshold=protocol['threshold'], source='weekly_library')
              for m in artifact['age_cases'][0]['versions'][0]['models']]
    plan = dict(schema_version=1, plan_kind='frozen_library_time_OE_IRL', config=config, irl_config=irl,
        config_source=dict(path=str(config_path), sha256=sha256_file(config_path)),
        irl_source=dict(path=str(irl_path), sha256=sha256_file(irl_path)),
        library_source=dict(path=str(library_path), sha256=sha256_file(library_path), plan_sha256=artifact['plan_sha256']),
        library_configuration=artifact['plan']['config'],
        cases=cases, policies=specs, code_sha256=experiment_code_hashes(),
        selection_policy='calibration_frozen_candidate_identity_and_weights_weekly_parameters_update',
        stage_roles=dict(train='bootstrap_history_only', calibration='expert_and_OE_reward_fit',
            validation='frozen_selection_diagnostics', test='predeclared_causal_updates_no_retuning'),
        holdout_claim='none_all_dates_are_development', age_selection='none')
    plan['plan_sha256'] = fingerprint(plan)
    return plan


def replay_candidates(reader, days, library_case, specs, protocol, *, detail=False):
    """一日一次生成 N×K 可见预测，每个候选独立相同成本回放。

    库的模型 ID 表示冻结的算法/特征/历史规则，参数可按周更新；不因验证净利
    改选。失败/未预热模型只在当下不可预测时禁用，不按未来标签有效性删决策。
    日账户重置、队列不跨日；缺尾未平仓保留并使汇总净利未定义。
    """
    daily, audits = [], []
    models = library_case['versions'][0]['models']
    indices = {m['model_id']: i for i, m in enumerate(models)}
    reference = library_case['fixed_reference'] | dict(features=FEATURE_COLUMNS, family='Ridge')
    fixed_id = 'fixed-' + fingerprint(reference)
    reward = IRLRewardLearner(horizons=protocol['reward_horizons_ms'], definition='paper_price_difference')
    for day in days:
        frame = read_day(reader, day)
        audit = audit_horizons(frame, reader.calendar, protocol['reward_horizons_ms'])
        audits.append(audit)
        matrix, ids = predict_library(library_case['versions'], frame, reader.report['interval_ms'])
        reference_preds = np.full(len(frame), np.nan)
        valid = frame.feature_valid.to_numpy()
        if reference['status'] == 'fitted' and valid.any():
            reference_preds[valid] = predict_model(reference, frame.loc[valid, FEATURE_COLUMNS])
        results = []
        for spec in specs:
            predictions = (np.zeros(len(frame)) if spec['cash'] else
                           reference_preds if spec['source'] == 'fixed_reference' else matrix[:, indices[spec['model_id']]])
            versions = ids if spec['source'] == 'weekly_library' else np.full(len(frame), 'cash' if spec['cash'] else fixed_id, dtype=object)
            quotes = frame[SNAPSHOT_SCHEMA.names + ['session_id', 'segment_id', 'mid_price', 'feature_valid']].copy()
            # 只增加当时模型可预测性条件；未来标签/未来成熟情况不在掩码中。
            quotes['feature_valid'] = valid & np.isfinite(predictions)
            model_name = reference['name'] if spec['source'] == 'fixed_reference' else spec['model_id']
            selector = (FlatSelector if spec['cash'] else SingleModelSelector)(spec['policy_id'], [SimpleNamespace(name=model_name)])
            selector.reward_type = 'OE'
            engine = TimeExecutionEngine(reward, interval_ms=reader.report['interval_ms'], calendar=reader.calendar,
                latency_ms=protocol['latency_ms'], holding_review_ms=protocol['holding_review_ms'],
                threshold=spec['threshold'], force_replay_end=False)
            result = engine.run_backtest(selector, quotes, predictions[:, None], detail=detail, model_version_ids=versions)
            result['session_boundary'] = dict(policy='independent_daily_accounts_no_feedback_carry',
                unexecuted_intents_not_forwarded=result['unexecuted_intents_at_end'],
                unmatured_OE_not_forwarded=result['unmatured_fill_rewards_at_end'],
                terminal_position=result['terminal_position'],
                unsettled_net_pnl_cannot_aggregate=not result['terminal_position_liquidated'])
            results.append(result)
        daily.append(dict(session_id=day, input_feature_valid_rows=int(valid.sum()),
            library_version_ids=sorted(set(ids) - {None}), results=results))
    return daily, audits


def frozen_evaluations(statistics, fits, baseline_id):
    """直接列出校准冻结选择的实际成交/费用/净利，不用评价期最佳候选替代。

    等权选择、校准净利专家、现金和原门槛基线都保留。不可用拟合为 None；
    现金选择净利可为零，但实测成熟订单 OE 仍未定义。
    """
    lookup = {r['policy_id']: r for r in statistics}
    first = next(iter(fits.values()))
    selected = {'cash': 'cash', 'fixed_reference_x1': baseline_id,
        'calibration_net_expert': first['expert']['selected_policy_id'] if first['expert'] else None,
        'equal_weight': first['equal_weight_selected_policy']['selected_policy_id'] if first['equal_weight_selected_policy'] else None}
    selected |= {name: f['selected_policy']['selected_policy_id'] if f['selected_policy'] else None for name, f in fits.items()}
    return {name: dict(selected_policy_id=identity, result=lookup[identity] if identity else None,
                       selection_source='predeclared_baseline' if name in ('cash', 'fixed_reference_x1') else 'calibration_only')
            for name, identity in selected.items()}


def run_library_oe(plan_path, *, detail=False):
    """先校准再评价；只冻结选择/权重，允许冻结规则定义的因果周训练。"""
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if (plan.get('schema_version') != 1 or plan.get('plan_kind') != 'frozen_library_time_OE_IRL'
            or plan.get('plan_sha256') != fingerprint(plan)):
        raise ValueError('Frozen library OE plan integrity check failed')
    if plan['code_sha256'] != experiment_code_hashes():
        raise ValueError('Source changed after freezing')
    source = plan['library_source']
    if sha256_file(source['path']) != source['sha256']:
        raise ValueError('Library artifact changed after freezing')
    artifact, calendar = load_library(source['path'])
    if artifact['plan']['config'] != plan['library_configuration']:
        raise ValueError('Library configuration changed after freezing')
    cases = []
    with threadpool_limits(limits=1):
        for case in plan['cases']:
            binding = case['session_binding']
            if binding['plan_sha256'] != fingerprint(binding):
                raise ValueError('Session binding changed')
            protocol = binding['protocol']
            reader = PreparedDatasetReader(binding['dataset']['path'], calendar=calendar,
                allow_partial=protocol['allow_partial'], include_degraded=protocol['include_degraded'])
            if reader.provenance != binding['dataset']:
                raise ValueError('Prepared dataset changed after freezing')
            library_case = next(c for c in artifact['age_cases'] if c['max_age_ms'] == case['max_age_ms'])
            daily, audits = replay_candidates(reader, protocol['sessions']['calibration'], library_case, plan['policies'], protocol, detail=detail)
            statistics = pooled_policy_statistics(daily, plan['policies'], protocol['reward_horizons_ms'])
            fits = {name: learn_calibration_reward(statistics, protocol['reward_horizons_ms'], plan['irl_config'], name)
                    for name in plan['irl_config']['weight_constraints']}
            phases = dict(calibration=dict(daily=daily, statistics=statistics, horizon_audits=audits))
            baseline_id = next(s['policy_id'] for s in plan['policies']
                               if s['source'] == 'fixed_reference' and s['policy_id'] == 'Ridge-threshold-x1')
            for phase in ('validation', 'test'):
                daily, audits = replay_candidates(reader, protocol['sessions'][phase], library_case, plan['policies'], protocol, detail=detail)
                statistics = evaluate_frozen_weights(pooled_policy_statistics(daily, plan['policies'], protocol['reward_horizons_ms']), fits)
                phases[phase] = dict(daily=daily, statistics=statistics, horizon_audits=audits,
                    frozen_evaluations=frozen_evaluations(statistics, fits, baseline_id))
            cases.append(dict(max_age_ms=case['max_age_ms'], fixed_reference=library_case['fixed_reference'],
                library_versions=[{k: v for k, v in version.items() if k != 'models'} for version in library_case['versions']],
                reward_fits=fits, phases=phases))
            print(f'完成 {case["max_age_ms"]}ms：' + ', '.join(f'{name}={fit["status"]}' for name, fit in fits.items()), flush=True)
    if plan['code_sha256'] != experiment_code_hashes() or sha256_file(source['path']) != source['sha256']:
        raise ValueError('Code/library changed during experiment')
    return dict(schema_version=1, result_kind='library_time_OE_IRL', plan=plan, plan_sha256=plan['plan_sha256'],
        age_cases=cases, selected_age_ms=None,
        environment=dict(python=platform.python_version(), **{name: importlib.metadata.version(name) for name in
            ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'scipy', 'threadpoolctl')}),
        instrument=INSTRUMENT_CONFIG['CME_ES'],
        limits=['Algorithm 1 使用有限候选最大间隔近似，尚无 Algorithm 2/3 或时间 ME IRL。',
                '校准冻结候选身份/权重；模型参数可按预声明周规则更新，不称数值模型全程冻结。',
                '每日独立账户，不跨日迁移意图/反馈；缺尾未平仓保留且净利汇总未定义。',
                'OE 只评价完整成熟订单子集，现金零向量仅为优化约定，不是实测零 OE。',
                '权重不唯一可识别；求解/可表示性/盈利不同，全部年龄和负结果保留。',
                '模型特征/短历史/树抽样仍是工程近似，全部日期用于开发。'])


def render_library_oe(result):
    """同一 JSON 展示校准状态、冻结选择和全部候选成本，避免只展示获选策略。"""
    lines = ['# 多模型时间 OE 校准与冻结选择评估', '', f"计划：`{result['plan_sha256']}`；全部年龄保留。", '',
        '| 年龄 ms | 权约束 | 状态 | 专家 | 冻结身份 | 正利润交易专家 | 权重 |',
        '| ---: | --- | --- | --- | --- | --- | --- |']
    for case in result['age_cases']:
        for name, fit in case['reward_fits'].items():
            lines.append(f"| {case['max_age_ms']} | {name} | {fit['status']} | "
                f"{fit['expert']['selected_policy_id'] if fit['expert'] else '未定义'} | "
                f"{fit['selected_policy']['selected_policy_id'] if fit['selected_policy'] else '未定义'} | "
                f"{fit.get('profitable_trading_expert', False)} | {fit['weights']} |")
    lines += ['', '## 冻结选择的实际结果', '',
        '| 年龄 ms | 阶段 | 对照/约束 | 冻结身份 | 成交 | 成熟 OE | 净利 USD |',
        '| ---: | --- | --- | --- | ---: | ---: | --- |']
    for case in result['age_cases']:
        for phase in ('validation', 'test'):
            for name, evaluation in case['phases'][phase]['frozen_evaluations'].items():
                row = evaluation['result']
                lines.append(f"| {case['max_age_ms']} | {phase} | {name} | {evaluation['selected_policy_id']} | "
                    f"{row['total_fills'] if row else '未定义'} | {row['matured_order_count'] if row else '未定义'} | "
                    f"{row['net_pnl_usd'] if row else '未定义'} |")
    lines += ['', '## 全候选订单覆盖与成本', '',
        '| 年龄 ms | 阶段 | 候选 | 成交 | 成熟 OE | 毛利 USD | 成本 USD | 净利 USD |',
        '| ---: | --- | --- | ---: | ---: | --- | --- | --- |']
    for case in result['age_cases']:
        for phase, data in case['phases'].items():
            for row in data['statistics']:
                lines.append(f"| {case['max_age_ms']} | {phase} | {row['policy_id']} | {row['total_fills']} | "
                    f"{row['matured_order_count']} | {row['gross_pnl_usd']} | {row['friction_usd']} | {row['net_pnl_usd']} |")
    lines += ['', '## 校准可表示性和失败诊断', '']
    for case in result['age_cases']:
        lines += [f"### {case['max_age_ms']}ms", '', '```json', pretty_json(case['reward_fits']), '```', '']
    lines += ['## 复现限制', ''] + ['- ' + s for s in result['limits']]
    return '\n'.join(lines) + '\n'
