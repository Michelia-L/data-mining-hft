"""冻结并比较全候选/库内专家的 OE 校准口径，另存结果且不执行动态交易。"""
import argparse
from pathlib import Path

from src.artifacts import pretty_json
from src.expert_scope import freeze_comparison, render_comparison, run_comparison
from src.session_experiment import publish_bundle


def main(argv=None):
    """先绑定归档字节与规则，再运行计划；已存在目录拒绝覆盖以保留旧结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--config', required=True); freeze.add_argument('--input', required=True)
    run = commands.add_parser('run'); run.add_argument('--plan', required=True)
    for command in (freeze, run):
        command.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        raise FileExistsError('Output directory must not exist')
    if args.command == 'freeze':
        result = freeze_comparison(args.config, args.input)
        report = '# 冻结专家来源对照\n\n尚未计算新专家或权重。\n\n```json\n' + pretty_json(result) + '\n```\n'
        publish_bundle(result, report, args.output_dir, 'plan.json')
    else:
        result = run_comparison(args.plan)
        publish_bundle(result, render_comparison(result), args.output_dir, 'result.json')
    print(f'已保存 {args.command} 产物：{args.output_dir}')
    return result


if __name__ == '__main__':
    main()
