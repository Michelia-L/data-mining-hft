"""先冻结既有校准与后段范围，再完整执行27项状态和最小赋权消融。"""
import argparse
from pathlib import Path

from src.artifacts import pretty_json
from src.frozen_ablation import freeze_ablation, plot_ablation, render_ablation, run_ablation
from src.session_experiment import publish_bundle


def main(argv=None):
    """freeze接收新数据/旧证据；run只消费冻结计划，另存结果保护历史产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--config', required=True)
    freeze.add_argument('--prior-evidence', required=True)
    freeze.add_argument('--prior-result', required=True)
    freeze.add_argument('--dataset-dirs', nargs='+', required=True)
    run = commands.add_parser('run')
    run.add_argument('--plan', required=True)
    run.add_argument('--checkpoint-dir', required=True)
    run.add_argument('--detail', action='store_true')
    for command in (freeze, run):
        command.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        raise FileExistsError('Output directory must not exist')
    if args.command == 'freeze':
        result = freeze_ablation(args.config, args.prior_evidence, args.prior_result, args.dataset_dirs)
        report = '# 冻结后段复核计划\n\n收益运行前保存；不重学奖励。\n\n```json\n'+pretty_json(result)+'\n```\n'
        publish_bundle(result, report, args.output_dir, 'plan.json')
    else:
        result = run_ablation(args.plan, checkpoint_dir=args.checkpoint_dir, detail=args.detail)
        publish_bundle(result, render_ablation(result), args.output_dir, 'result.json')
        plot_ablation(result, args.output_dir)
    print(f'已保存 {args.command} 产物：{args.output_dir}', flush=True)
    return result


if __name__ == '__main__':
    main()
