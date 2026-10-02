"""冻结全部年龄与有限策略，按时间 OE 学习奖励并评估冻结策略。"""
import argparse
from pathlib import Path

from src.artifacts import pretty_json
from src.session_experiment import publish_bundle
from src.snapshot_dataset import DEFAULT_CALENDAR
from src.snapshot_irl import freeze_time_irl, render_time_irl, run_time_irl


def main(argv=None):
    """run 只消费冻结计划；修改候选、日期或约束必须另存配置和计划。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--config', required=True)
    freeze.add_argument('--dataset-dirs', nargs='+', required=True)
    freeze.add_argument('--calendar', default=str(DEFAULT_CALENDAR))
    run = commands.add_parser('run')
    run.add_argument('--plan', required=True)
    run.add_argument('--detail', action='store_true')
    for command in (freeze, run):
        command.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        raise FileExistsError('Output directory must not exist')
    if args.command == 'freeze':
        result = freeze_time_irl(args.config, args.dataset_dirs, args.calendar)
        report = '# 冻结时间尺度 IRL 计划\n\n冻结时不选专家或查看收益。\n\n```json\n' + pretty_json(result) + '\n```\n'
        publish_bundle(result, report, args.output_dir, 'plan.json')
    else:
        result = run_time_irl(args.plan, detail=args.detail)
        publish_bundle(result, render_time_irl(result), args.output_dir, 'result.json')
    print(f'已保存 {args.command} 产物：{args.output_dir}')
    return result


if __name__ == '__main__':
    main()
