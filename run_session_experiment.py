"""先冻结完整 session 的四段开发协议，再消费冻结文件运行固定模型基线。"""
import argparse
from pathlib import Path

from src.session_experiment import (freeze_protocol, publish_bundle, render_plan,
                                    render_result, run_frozen_protocol)
from src.snapshot_dataset import DEFAULT_CALENDAR


def main(argv=None):
    """run 不接受日期、门槛、模型或奖励参数覆盖，避免测试过程中静默换协议。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze', help='检查配置与来源并保存绑定哈希的计划')
    freeze.add_argument('--protocol', required=True, help='四段 session 与固定参数 JSON')
    freeze.add_argument('--dataset-dir', required=True, help='prepared snapshots 与 quality 目录')
    freeze.add_argument('--calendar', default=str(DEFAULT_CALENDAR), help='与 prepared 一致的日历')
    freeze.add_argument('--output-dir', required=True, help='尚不存在的计划目录')
    run = commands.add_parser('run', help='按冻结计划逐日训练/审计/回放，不覆盖实验参数')
    run.add_argument('--plan', required=True, help='冻结的 plan.json')
    run.add_argument('--output-dir', required=True, help='尚不存在的结果目录')
    run.add_argument('--detail', action='store_true', help='保存逐笔成交、反馈与资金审计')
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        raise FileExistsError('Output directory must not exist')
    if args.command == 'freeze':
        result = freeze_protocol(args.protocol, args.dataset_dir, calendar_path=args.calendar)
        publish_bundle(result, render_plan(result), args.output_dir, 'plan.json')
    else:
        result = run_frozen_protocol(args.plan, detail=args.detail)
        publish_bundle(result, render_result(result), args.output_dir, 'result.json')
    print(f'已保存 {args.command} 产物：{args.output_dir}')
    return result


if __name__ == '__main__':
    main()
