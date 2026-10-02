"""冻结并构建按周轻模型库，另存模型参数和全候选预测诊断。"""
import argparse
from pathlib import Path

from src.artifacts import pretty_json
from src.session_experiment import publish_bundle
from src.snapshot_dataset import DEFAULT_CALENDAR
from src.weekly_library import build_library, freeze_library, render_library


def main(argv=None):
    """输出目录须不存在；build 只消费冻结计划，不接受收益驱动的参数覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--config', required=True)
    freeze.add_argument('--dataset-dirs', nargs='+', required=True)
    freeze.add_argument('--calendar', default=str(DEFAULT_CALENDAR))
    build = commands.add_parser('build')
    build.add_argument('--plan', required=True)
    for command in (freeze, build):
        command.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        raise FileExistsError('Output directory must not exist')
    if args.command == 'freeze':
        result = freeze_library(args.config, args.dataset_dirs, args.calendar)
        report = '# 冻结按周模型库计划\n\n不计算收益或选择参数。\n\n```json\n' + pretty_json(result) + '\n```\n'
        publish_bundle(result, report, args.output_dir, 'plan.json')
    else:
        result = build_library(args.plan)
        publish_bundle(result, render_library(result), args.output_dir, 'library.json')
    print(f'已保存 {args.command} 产物：{args.output_dir}')
    return result


if __name__ == '__main__':
    main()
