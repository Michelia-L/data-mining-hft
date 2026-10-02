"""固定 OE 比较集合，只改变 Algorithm 1 工程专家的来源范围。

物理第3页 Eq.(1) 将动作对应到模型，第4页 Algorithm 1 只说明从训练回测
取得专家，未公开筛选规则。因此全候选/库内候选均为显式工程对照。这里
复核归档校准账本，不重放交易、不读取后段指标、不自动激活求出的权重。
"""
import importlib.metadata
import json
from pathlib import Path
import platform

from threadpoolctl import threadpool_limits

from src.config import BASE_DIR
from src.learned_oe_experiment import execution_gate
from src.library_oe import experiment_code_hashes
from src.session_experiment import fingerprint
from src.snapshot_dataset import sha256_file
from src.snapshot_irl import learn_calibration_reward, pooled_policy_statistics, validate_irl_config
from src.sum_only_reward import learn_sum_only

SCOPES = ('all_candidates', 'online_library')
CONSTRAINTS = ('simplex', 'signed_box', 'sum_only')


def code_hashes():
    """新诊断绑定当前实现；归档保留旧源码身份，不能冒称当前代码重放结果。"""
    return experiment_code_hashes() | {p: sha256_file(BASE_DIR / p) for p in
        ('src/learned_oe_experiment.py', 'src/sum_only_reward.py',
         'src/expert_scope.py', 'run_expert_scope.py')}


def validate_config(config):
    """两个来源、三个权重约束全部预声明，禁止仅保留成功的来源或报价年龄。"""
    keys = {'schema_version', 'purpose', 'expert_scopes', 'weight_constraints', 'oracle_scope'}
    if (set(config) - keys - {'notes'} or keys - set(config) or config['schema_version'] != 1
            or config['purpose'] != 'development_OE_expert_scope_comparison'
            or config['expert_scopes'] != list(SCOPES)
            or config['weight_constraints'] != list(CONSTRAINTS)
            or config['oracle_scope'] != 'all_eligible_calibration_candidates'):
        raise ValueError('Need both expert scopes and the unchanged full comparison set')


def fit_scope(statistics, horizons, config, policies, scope):
    """输入同一个 P×H 价格点均值矩阵；仅专家候选来源不同。

    库内规则先从全部已结算库策略中按净利选专家，包括负净利和无成熟订单
    策略。若最佳者没有 OE，仍阻断。现金/固定 Ridge 继续参与有限策略响应
    优化，不能通过删掉难以击败的外部参照来制造可表示性。在线门控始终针对
    原来的完整模型库，允许求解和通过门控并不等于专家盈利或权重唯一识别。
    """
    if scope not in SCOPES:
        raise ValueError('Unknown expert scope')
    names = [p['policy_id'] for p in policies]
    if names != [r['policy_id'] for r in statistics] or len(set(names)) != len(names):
        raise ValueError('Statistics must retain every declared policy in order')
    experts = names if scope == 'all_candidates' else [
        p['policy_id'] for p in policies if p['source'] == 'weekly_library']
    fits = {c: learn_calibration_reward(statistics, horizons, config, c, expert_policy_ids=experts)
            for c in CONSTRAINTS[:-1]}
    fits['sum_only'] = learn_sum_only(statistics, horizons, config, fits['signed_box'])
    return dict(expert_scope=scope, expert_candidate_policy_ids=experts, reward_fits=fits,
        execution_gates={c: execution_gate(f, policies) for c, f in fits.items()},
        active_reward_weights=None)


def calibration_payload(input_path):
    """从多周归档只抽取校准账本，重新按成熟订单数合并所有候选。

    校验父计划、嵌套会话身份、所有年龄与每日候选完整性。全归档字节绑定
    用于追溯；后段预测质量/收益/就绪标志均不参与本次专家或权重计算。
    原拟合另存，用于运行时核对全候选控制完全重现，不能静默替换旧基线。
    """
    artifact = json.loads(Path(input_path).read_text(encoding='utf-8'))
    if artifact.get('schema_version') != 1 or artifact.get('result_kind') != 'multiweek_OE_readiness_development':
        raise ValueError('Need an archived multiweek readiness result')
    plan = artifact['plan']; base = plan['base_oe_plan']
    if artifact['plan_sha256'] != plan['plan_sha256']:
        raise ValueError('Archived result plan identity changed')
    for value, kind in ((plan, 'frozen_multiweek_OE_readiness'), (base, 'frozen_library_time_OE_IRL')):
        if value.get('plan_kind') != kind or value['plan_sha256'] != fingerprint(value):
            raise ValueError('Archived source plan integrity check failed')
    config = base['irl_config']; validate_irl_config(config)
    if config['weight_constraints'] != list(CONSTRAINTS[:-1]):
        raise ValueError('Archive must retain both bounded controls')
    ages = config['age_candidates_ms']
    if ([c['max_age_ms'] for c in artifact['age_cases']] != ages
            or [c['max_age_ms'] for c in base['cases']] != ages):
        raise ValueError('Every declared age case must remain in order')
    policies = base['policies']; names = [p['policy_id'] for p in policies]
    if len(set(names)) != len(names) or not any(p['source'] == 'weekly_library' for p in policies):
        raise ValueError('Need unique policies including the online library')
    cases = []
    for case, frozen in zip(artifact['age_cases'], base['cases']):
        binding = frozen['session_binding']; protocol = binding['protocol']
        if binding['plan_sha256'] != fingerprint(binding):
            raise ValueError('Archived session binding changed')
        cal = case['calibration']
        if [d['session_id'] for d in cal['daily']] != protocol['sessions']['calibration']:
            raise ValueError('Only declared calibration sessions may define expectations')
        if any([r['strategy'] for r in d['results']] != names for d in cal['daily']):
            raise ValueError('Daily ledger must retain every candidate exactly once in order')
        stats = pooled_policy_statistics(cal['daily'], policies, protocol['reward_horizons_ms'])
        if stats != cal['statistics']:
            raise ValueError('Calibration summary disagrees with daily ledger')
        if case['calibration_sha256'] != fingerprint(dict(statistics=stats,
                reward_fits=case['reward_fits'], execution_gate=case['execution_gate'])):
            raise ValueError('Archived calibration identity changed')
        cases.append(dict(max_age_ms=case['max_age_ms'], statistics=stats,
            horizons_ms=protocol['reward_horizons_ms'], historical_session_binding=binding,
            historical_reward_fits=case['reward_fits'], historical_execution_gate=case['execution_gate']))
    return dict(archive_plan_sha256=plan['plan_sha256'], archive_code_sha256=plan['code_sha256'],
        irl_config=config, policies=policies, cases=cases)


def freeze_comparison(config_path, input_path):
    """冻结已用开发校准段、两组来源和完整候选，尚不计算新专家或权重。"""
    config_path = Path(config_path).resolve(); input_path = Path(input_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8')); validate_config(config)
    digest = sha256_file(input_path); payload = calibration_payload(input_path)
    if digest != sha256_file(input_path):
        raise ValueError('Archive changed during freeze')
    plan = dict(schema_version=1, plan_kind='frozen_OE_expert_scope_comparison', config=config,
        config_source=dict(path=str(config_path), sha256=sha256_file(config_path)),
        archived_result_source=dict(path=str(input_path), sha256=digest), calibration=payload,
        code_sha256=code_hashes(), selected_age_ms=None, selected_expert_scope=None,
        holdout_claim='none_existing_development_calibration',
        activation_policy='diagnostic_only_no_dynamic_execution')
    plan['plan_sha256'] = fingerprint(plan)
    return plan


def run_comparison(plan_path):
    """复算旧控制再拟合新来源；身份、统计或基线不一致时拒绝发布结果。"""
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if (plan.get('schema_version') != 1 or plan.get('plan_kind') != 'frozen_OE_expert_scope_comparison'
            or plan.get('plan_sha256') != fingerprint(plan) or plan['code_sha256'] != code_hashes()):
        raise ValueError('Frozen comparison plan/source integrity check failed')
    validate_config(plan['config']); source = plan['archived_result_source']
    if sha256_file(source['path']) != source['sha256']:
        raise ValueError('Archived input changed after freezing')
    payload = calibration_payload(source['path'])
    if payload != plan['calibration']:
        raise ValueError('Frozen calibration payload changed')
    cases = []
    with threadpool_limits(limits=1):
        for case in payload['cases']:
            groups = {s: fit_scope(case['statistics'], case['horizons_ms'], payload['irl_config'],
                                  payload['policies'], s) for s in SCOPES}
            control = groups['all_candidates']
            if (control['reward_fits'] != case['historical_reward_fits'] or
                    control['execution_gates']['sum_only'] != case['historical_execution_gate']):
                raise ValueError('Full-candidate control differs from archived calibration')
            # 同一个成熟资格集合意味着两组面对相同 oracle，而不是只同名的不同矩阵。
            if any(control['reward_fits'][c]['eligible_policy_ids'] !=
                    groups['online_library']['reward_fits'][c]['eligible_policy_ids'] for c in CONSTRAINTS):
                raise ValueError('Expert comparison changed the optimization candidate set')
            cases.append(dict(max_age_ms=case['max_age_ms'], scopes=groups,
                full_candidate_control_reproduced=True, common_statistics_sha256=fingerprint(
                    dict(statistics=case['statistics'])), dynamic_backtest_run=False))
            print(f"完成 {case['max_age_ms']}ms 两种专家来源、三个约束对照", flush=True)
    if code_hashes() != plan['code_sha256'] or sha256_file(source['path']) != source['sha256']:
        raise ValueError('Source/archive changed during comparison')
    return dict(schema_version=1, result_kind='OE_expert_scope_comparison', plan=plan,
        plan_sha256=plan['plan_sha256'], age_cases=cases, selected_age_ms=None,
        selected_expert_scope=None, formal_replication_ready=False,
        environment=dict(python=platform.python_version(), **{n: importlib.metadata.version(n)
            for n in ('numpy', 'pandas', 'scipy', 'threadpoolctl')}),
        limits=['专家来源两种口径均为工程假设，原文没有公开专家筛选规则。',
            '同一完整有限策略矩阵近似 Algorithm 1，sum_only 第二阶段最小 L1 仅作消歧。',
            '只复核归档开发校准账本；没有以当前代码重放交易或新增未触碰测试集。',
            '仅成熟订单子集有 OE；现金零向量仍为优化约定，负净利专家不称盈利。',
            '门控通过仅表示具备后续动态实验的校准条件；未激活任何权重或选择来源/年龄。'])


def render_comparison(result):
    """全部18个结果同表展示，失败、负净利、并列专家和权重都保留在 JSON。"""
    lines = ['# OE 专家来源校准对照', '', f"冻结计划 `{result['plan_sha256']}`；仅开发诊断。", '',
        '| 年龄 ms | 专家来源 | 专家 | 校准净利 USD | 约束 | 拟合状态 | 门控通过 | 原因 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for case in result['age_cases']:
        for name, group in case['scopes'].items():
            for constraint, fit in group['reward_fits'].items():
                expert = fit['expert'] or {}; gate = group['execution_gates'][constraint]
                lines.append(f"| {case['max_age_ms']} | {name} | {expert.get('selected_policy_id')} | "
                    f"{expert.get('best_score')} | {constraint} | {fit['status']} | "
                    f"{gate['usable_for_execution']} | {gate['reasons']} |")
    lines += ['', '门控通过不代表动态收益、盈利专家或正式复现成功；全部权重未激活。', '',
              '## 限制', ''] + ['- ' + s for s in result['limits']]
    return '\n'.join(lines) + '\n'
