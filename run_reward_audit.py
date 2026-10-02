"""冻结并审计历史校准 OE IRL 的专家线性可表示性。"""
import argparse
from pathlib import Path

from src.artifacts import pretty_json
from src.reward_audit_experiment import freeze_reward_audit, render_reward_audit, run_reward_audit
from src.session_experiment import publish_bundle


def main(argv=None):
    """输出只能新建，保留输入归档和旧结果；不提供权重激活或收益选参选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--config', required=True);freeze.add_argument('--input', required=True)
    run = commands.add_parser('run');run.add_argument('--plan', required=True)
    for command in (freeze, run):command.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():raise FileExistsError('Output directory must not exist')
    if args.command == 'freeze':
        result = freeze_reward_audit(args.config, args.input)
        report = '# 冻结可表示性审计计划\n\n尚未求解，输入仅为历史校准数据。\n\n```json\n'+pretty_json(result)+'\n```\n'
        publish_bundle(result, report, args.output_dir, 'plan.json')
    else:
        result = run_reward_audit(args.plan)
        publish_bundle(result, render_reward_audit(result), args.output_dir, 'result.json')
    print(f'已保存 {args.command} 产物：{args.output_dir}')
    return result


if __name__ == '__main__':main()
