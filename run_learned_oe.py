"""冻结并运行校准OE权重与UCB/双ARS联动，保留失败与原18个控制。"""
import argparse
from pathlib import Path

from src.artifacts import pretty_json
from src.learned_oe_experiment import freeze_learned_experiment, render_learned_experiment, run_learned_experiment
from src.session_experiment import publish_bundle


def main(argv=None):
    """配置只在冻结阶段读取，运行不能覆盖参数；新建目录保护历史结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze'); freeze.add_argument('--config', required=True); freeze.add_argument('--library', required=True)
    run = commands.add_parser('run'); run.add_argument('--plan', required=True); run.add_argument('--detail', action='store_true')
    for command in (freeze, run): command.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists(): raise FileExistsError('Output directory must not exist')
    if args.command == 'freeze':
        result = freeze_learned_experiment(args.config, args.library)
        report = '# 冻结OE学习联动计划\n\n尚不计算收益。\n\n```json\n'+pretty_json(result)+'\n```\n'
        publish_bundle(result, report, args.output_dir, 'plan.json')
    else:
        result = run_learned_experiment(args.plan, detail=args.detail)
        publish_bundle(result, render_learned_experiment(result), args.output_dir, 'result.json')
    print(f'已保存 {args.command} 产物：{args.output_dir}')
    return result


if __name__ == '__main__': main()
