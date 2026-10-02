"""绑定历史多模型 OE 校准产物，仅诊断线性奖励的可表示性。

输入可来自旧代码的归档结果：保留当时源码/数据身份，不声称用当前执行器
重放过。这里只重算校准统计、旧拟合与几何证书，不消费验证或测试数值。
"""
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from threadpoolctl import threadpool_limits

from src.config import BASE_DIR
from src.library_oe import experiment_code_hashes
from src.reward_audit import CONSTRAINTS, audit_calibration
from src.session_experiment import fingerprint
from src.snapshot_dataset import sha256_file
from src.snapshot_irl import pooled_policy_statistics, validate_irl_config


def code_hashes():
    """绑定本次诊断实现和统计/旧拟合依赖，不覆盖归档实验的历史源码身份。"""
    return experiment_code_hashes() | {name: sha256_file(BASE_DIR/name) for name in
        ('src/reward_audit.py', 'src/reward_audit_experiment.py', 'run_reward_audit.py')}


def validate_config(config):
    """三个集合预声明保留，HiGHS 可接受的数值精度与证书验证精度分别冻结。"""
    required = {'schema_version', 'purpose', 'constraints', 'verification_tolerance',
                'solver_feasibility_tolerance'}
    if (set(config)-required-{'notes'} or required-set(config) or config['schema_version'] != 1
            or config['purpose'] != 'development_reward_representability_audit'
            or config['constraints'] != list(CONSTRAINTS)
            or any(type(config[k]) not in (int, float) or not np.isfinite(config[k]) or config[k] <= 0
                   for k in ('verification_tolerance', 'solver_feasibility_tolerance'))
            or not 1e-10 <= config['solver_feasibility_tolerance'] <= config['verification_tolerance']):
        raise ValueError('Predeclared constraint sets and valid verification tolerances required')


def calibration_payload(input_path):
    """校验归档身份和校准统计账本，只提取已声明校准日，不读后两段的统计。

    归档整个 JSON 的哈希用于追溯，包含后续段落并不代表它们用于选择。
    重新由校准每日成交统计合并均值，拒绝与其账本不一致的摘要；现金仍为
    无实测 OE，只有优化中使用零向量。旧源码无需等于当前源码，明确为归档诊断。
    """
    artifact = json.loads(Path(input_path).read_text(encoding='utf-8'))
    if artifact.get('schema_version') != 1 or artifact.get('result_kind') != 'library_time_OE_IRL':
        raise ValueError('Need an archived library_time_OE_IRL result')
    plan = artifact['plan']
    if artifact['plan_sha256'] != plan['plan_sha256'] or plan['plan_sha256'] != fingerprint(plan):
        raise ValueError('Archived source plan integrity check failed')
    validate_irl_config(plan['irl_config'])
    ages = plan['irl_config']['age_candidates_ms']
    if ([c['max_age_ms'] for c in artifact['age_cases']] != ages
            or [c['max_age_ms'] for c in plan['cases']] != ages):
        raise ValueError('Every archived age case must remain in order')
    cases = []
    for case, frozen in zip(artifact['age_cases'], plan['cases']):
        binding = frozen['session_binding'];protocol = binding['protocol']
        if binding['plan_sha256'] != fingerprint(binding):
            raise ValueError('Archived session binding changed')
        calibration = case['phases']['calibration']
        if [d['session_id'] for d in calibration['daily']] != protocol['sessions']['calibration']:
            raise ValueError('Only declared calibration sessions may define expectations')
        statistics = pooled_policy_statistics(calibration['daily'], plan['policies'], protocol['reward_horizons_ms'])
        if calibration['statistics'] != statistics:
            raise ValueError('Calibration summary disagrees with its daily ledger statistics')
        cases.append(dict(max_age_ms=case['max_age_ms'], horizons_ms=protocol['reward_horizons_ms'],
            calibration_sessions=protocol['sessions']['calibration'], statistics=statistics,
            historical_session_binding=binding))
    return dict(archive_plan_sha256=artifact['plan_sha256'], archive_code_sha256=plan['code_sha256'],
        irl_config=plan['irl_config'], policies=plan['policies'], cases=cases,
        holdout_claim='none_diagnostics_of_existing_development_calibration',
        execution='none_no_replay_or_weight_activation')


def freeze_reward_audit(config_path, input_path):
    """冻结历史输入字节、全部校准候选与数值协议，不提前计算或选择证书。"""
    config_path, input_path = Path(config_path).resolve(), Path(input_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'));validate_config(config)
    before = sha256_file(input_path)
    payload = calibration_payload(input_path)
    if before != sha256_file(input_path):
        raise ValueError('Archived result changed during freeze')
    plan = dict(schema_version=1, plan_kind='frozen_reward_representability_audit', config=config,
        config_source=dict(path=str(config_path), sha256=sha256_file(config_path)),
        archived_result_source=dict(path=str(input_path), sha256=before),
        calibration=payload, code_sha256=code_hashes(), selected_age_ms=None,
        analysis_scope='calibration_statistics_only_all_ages_and_constraints',
        activation_policy='diagnostic_weights_never_used_for_execution')
    plan['plan_sha256'] = fingerprint(plan)
    return plan


def run_reward_audit(plan_path):
    """仅以冻结校准矩阵求证书，旧模型、旧静态交易结果和专家均不修改。"""
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if (plan.get('schema_version') != 1 or plan.get('plan_kind') != 'frozen_reward_representability_audit'
            or plan.get('plan_sha256') != fingerprint(plan) or plan['code_sha256'] != code_hashes()):
        raise ValueError('Frozen audit plan/source integrity check failed')
    validate_config(plan['config'])
    source = plan['archived_result_source']
    if sha256_file(source['path']) != source['sha256']:
        raise ValueError('Archived input changed after freezing')
    payload = calibration_payload(source['path'])
    if payload != plan['calibration']:
        raise ValueError('Frozen calibration payload changed')
    cases = []
    with threadpool_limits(limits=1):
        for case in payload['cases']:
            audit = audit_calibration(case['statistics'], case['horizons_ms'], payload['irl_config'], plan['config'])
            selected = audit['expert']['selected_policy_id'] if audit['expert'] else None
            spec = next((p for p in payload['policies'] if p['policy_id'] == selected), None)
            audit.update(expert_source=spec['source'] if spec else None,
                expert_identity_in_library_policy_set=bool(spec and spec['source'] == 'weekly_library'))
            cases.append(dict(max_age_ms=case['max_age_ms'], horizons_ms=case['horizons_ms'],
                calibration_sessions=case['calibration_sessions'], audit=audit))
            print(f"完成 {case['max_age_ms']}ms 校准可表示性审计：{audit['diagnosis']}", flush=True)
    if code_hashes() != plan['code_sha256'] or sha256_file(source['path']) != source['sha256']:
        raise ValueError('Source/archive changed during audit')
    return dict(schema_version=1, result_kind='reward_representability_audit', plan=plan,
        plan_sha256=plan['plan_sha256'], age_cases=cases, selected_age_ms=None,
        environment=dict(python=platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ('numpy', 'pandas', 'scipy', 'threadpoolctl')}),
        limits=['sum(w)=1 为 Eq.(3) 原文条件；非负/盒约束是项目附加条件。',
            '仅在已有校准成熟订单子集上诊断；没有新增实测订单或未触碰测试集。',
            '最小 L1 见证不是 Algorithm 1 最大间隔目标，不称奖励唯一识别或学习成功。',
            '不可行以独立数值证书复核；无法验证时结论为未定义，不能只依赖求解状态。',
            '现金零向量是优化约定，现金专家身份不是当前按模型选择器的一个模型臂。',
            '历史输入保存当时源码/数据身份，本次不以当前执行器重新回放，也不激活权重。'])


def render_reward_audit(result):
    """同一 JSON 列出所有集合、失败和原拟合；不把几何可行写成交易收益。"""
    lines = ['# OE IRL 可表示性审计', '', f"冻结计划 `{result['plan_sha256']}`；仅使用归档校准数据。", '',
        '| 年龄 ms | 专家 | 诊断 | 约束 | 状态 | 权重 L1 | 最大绝对权重 | 已验证不可行证书 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for case in result['age_cases']:
        audit = case['audit'];expert = audit['expert']['selected_policy_id'] if audit['expert'] else None
        if not audit['audits']:
            lines.append(f"| {case['max_age_ms']} | {expert} | {audit['diagnosis']} | 全部 | {audit['status']} | None | None | None |")
        for constraint, row in audit['audits'].items():
            witness = row['witness'] or {};proof = row['infeasibility_certificate'] or {}
            lines.append(f"| {case['max_age_ms']} | {expert} | {audit['diagnosis']} | {constraint} | {row['status']} | "
                f"{witness.get('L1')} | {witness.get('max_abs_weight')} | {proof.get('verified')} |")
    lines += ['', '## 可重算证据与原拟合状态', '']
    for case in result['age_cases']:
        lines += [f"### {case['max_age_ms']}ms", '', '```json', json.dumps(case['audit'], ensure_ascii=False, indent=2), '```', '']
    lines += ['## 限制', ''] + ['- '+note for note in result['limits']]
    return '\n'.join(lines)+'\n'
