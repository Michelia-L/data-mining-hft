"""冻结多模型时间 OE 校准协议，评估校准后固定的候选身份和奖励权重。"""
import argparse
from pathlib import Path

from src.artifacts import pretty_json
from src.library_oe import freeze_library_oe, render_library_oe, run_library_oe
from src.session_experiment import publish_bundle


def main(argv=None):
    """模型产物先由周建库入口产生；run 不覆盖任何参数或已有输出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--config', required=True)
    freeze.add_argument('--library', required=True)
    run = commands.add_parser('run')
    run.add_argument('--plan', required=True)
    run.add_argument('--detail', action='store_true')
    for command in (freeze, run):
        command.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        raise FileExistsError('Output directory must not exist')
    if args.command == 'freeze':
        result = freeze_library_oe(args.config, args.library)
        report = '# 冻结多模型时间 OE 计划\n\n尚不计算收益。\n\n```json\n' + pretty_json(result) + '\n```\n'
        publish_bundle(result, report, args.output_dir, 'plan.json')
    else:
        result = run_library_oe(args.plan, detail=args.detail)
        publish_bundle(result, render_library_oe(result), args.output_dir, 'result.json')
    print(f'已保存 {args.command} 产物：{args.output_dir}')
    return result


if __name__ == '__main__':
    main()
