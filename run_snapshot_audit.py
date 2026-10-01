"""冻结采样年龄候选，生成缺格诊断与长期标签对照；不修改默认交易实验。"""
import argparse
from pathlib import Path

from src.session_experiment import publish_bundle
from src.snapshot_audit import compare_audits, freeze_audit, run_age_audit
from src.snapshot_dataset import DEFAULT_CALENDAR
from src.artifacts import pretty_json


def main(argv=None):
    """freeze 不看收益，run 只接受冻结候选，compare 要求全部候选齐备。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--config', required=True)
    freeze.add_argument('--data-index', required=True)
    freeze.add_argument('--calendar', default=str(DEFAULT_CALENDAR))
    run = commands.add_parser('run')
    run.add_argument('--plan', required=True)
    run.add_argument('--max-age-ms', required=True, type=int)
    compare = commands.add_parser('compare')
    compare.add_argument('--plan', required=True)
    compare.add_argument('--results', required=True, nargs='+')
    for command in (freeze, run, compare):
        command.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        raise FileExistsError('Output directory must not exist')
    if args.command == 'freeze':
        result = freeze_audit(args.config, args.data_index, args.calendar)
        report = '# 冻结的数据质量诊断计划\n\n不计算收益或选择候选。\n\n```json\n' + pretty_json(result) + '\n```\n'
        publish_bundle(result, report, args.output_dir, 'plan.json')
    elif args.command == 'run':
        result = run_age_audit(args.plan, args.max_age_ms, args.output_dir)
    else:
        result, report = compare_audits(args.plan, args.results)
        publish_bundle(result, report, args.output_dir, 'comparison.json')
    print(f'已保存 {args.command} 产物：{args.output_dir}')
    return result


if __name__ == '__main__':
    main()
