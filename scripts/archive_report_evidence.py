"""为课程复现报告保存已完成实验的紧凑证据，不重新训练或选择参数。

输入是 PR #19 的完整真实多周 JSON；输出保留全部年龄、专家来源、拟合、
阶段汇总和逐日账本摘要。庞大的期间明细仍由原结果文件哈希定位，不能把
本快照称为完整交易审计。原始数值直接投影，None 不替换为零。
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DAILY_FIELDS = (
    'strategy', 'run_status', 'total_trades', 'total_fills', 'gross_pnl_usd',
    'friction_usd', 'net_pnl_usd', 'pnl_aggregation_defined', 'terminal_position',
    'terminal_position_liquidated', 'holding_ms_mean', 'reward_type', 'reward_weights',
    'matured_order_count', 'unmatured_fill_rewards_at_end', 'cold_start_periods',
    'period_status_counts', 'history_status_counts', 'expert_scope',
)


def archive(result, source_sha256, source_commit):
    """投影已保存的报告证据；按实际源码内容验证研究基线，避免误标当前分支。

    source_commit 是产生原结果的 Git 提交，不是编写报告的提交。本报告分支
    可以早于尚待合并的研究分支；因此从 Git 对象读取原源码，而不要求当前
    工作区导入或重跑研究模块。各阶段的统计、门控和时序审计逐字段保留。
    """
    plan = result['plan']
    if (result['result_kind'] != 'multiweek_dynamic_OE_development'
            or result['plan_sha256'] != plan['plan_sha256']
            or not re.fullmatch(r'[0-9a-f]{40}', source_commit)):
        raise ValueError('需要原真实多周开发结果及完整研究提交 SHA')
    for path, expected in plan['code_sha256'].items():
        original = subprocess.run(['git', 'show', f'{source_commit}:{path}'],
                                  cwd=ROOT, capture_output=True, check=False)
        if original.returncode or hashlib.sha256(original.stdout).hexdigest() != expected:
            raise ValueError(f'研究提交与结果源码身份不符：{path}')
    ars = plan['base_ars_plan']; period = ars['base_period_plan']
    binding = period['cases'][0]['session_binding']
    cases = []
    for case in result['age_cases']:
        phases = {}
        for name, phase in case['phases'].items():
            daily = []
            for day in phase['daily']:
                # 保留实际逐日收益和审计摘要；不复制数十万次窗口回放明细。
                daily.append({k: day[k] for k in ('session_id', 'library_version_ids',
                    'input_feature_valid_rows', 'dynamic_audits', 'prediction_audit', 'calibration_sha256')}
                    | dict(results=[{k: row[k] for k in DAILY_FIELDS if k in row}
                                    for row in day['results']]))
            phases[name] = dict(statistics=phase['statistics'], daily=daily,
                                calibration_sha256=phase['calibration_sha256'])
        cases.append(dict(max_age_ms=case['max_age_ms'], calibration_sha256=case['calibration_sha256'],
            calibration={k: case['calibration'][k] for k in ('statistics', 'scopes', 'horizon_audits')},
            training_visibility_audits=case['training_visibility_audits'], phases=phases,
            dynamic_version_coverage=case['dynamic_version_coverage']))
    return dict(schema_version=1, evidence_kind='fmato_report_development_snapshot',
        source_result_sha256=source_sha256, source_commit=source_commit, plan_sha256=result['plan_sha256'],
        source_code_sha256=plan['code_sha256'], environment=result['environment'], instrument=result['instrument'],
        formal_replication_ready=result['formal_replication_ready'], holdout_claim=plan['holdout_claim'],
        selected_age_ms=result['selected_age_ms'], selected_expert_scope=result['selected_expert_scope'],
        strategies=plan['strategies'], configuration=dict(dynamic=plan['config'], ars=ars['config'],
            period=period['config'], session=binding['protocol'], library=period['library_configuration']),
        expected_evaluation_versions=plan['expected_evaluation_versions'],
        library_source=period['library_source'], dataset_bindings=[dict(max_age_ms=c['max_age_ms'],
            dataset=c['session_binding']['dataset'], calendar=c['session_binding']['calendar'],
            source_coverage=c['session_binding']['source_coverage']) for c in period['cases']],
        age_cases=cases, limits=result['limits'])


def main():
    """核对原文件字节哈希后另存快照；不覆盖已归档的报告证据。

    只依赖 Python 标准库和本地 Git，查看快照不需要原始行情或项目数值环境。
    输出路径和原实验结果分开；可用新临时路径复核确定性的序列化结果。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('报告证据输出已存在，请另选新路径')
    raw = Path(args.input).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != args.expected_sha256:
        raise ValueError('原实验文件字节哈希不符，拒绝归档')
    value = archive(json.loads(raw), digest, args.source_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        # 快照是表格/图表的机器证据，紧凑保存；人工查看可用 python -m json.tool。
        # 只移除排版空白，既不缩减字段，也不舍入浮点或改写 null。
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        handle.write('\n')
    print(f'已保存报告证据：{output}')


if __name__ == '__main__':
    main()
