"""冻结并运行真实多周OE动态对照，保留两种专家来源和全部原控制。"""
import argparse
from pathlib import Path

from src.artifacts import pretty_json
from src.multiweek_dynamic_oe import freeze_dynamic, render_dynamic, run_dynamic
from src.session_experiment import publish_bundle


def main(argv=None):
    """只在freeze接受配置；run不覆盖参数，新输出目录保护历史结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--config', required=True); freeze.add_argument('--library', required=True)
    run = commands.add_parser('run')
    run.add_argument('--plan', required=True); run.add_argument('--detail', action='store_true')
    run.add_argument('--checkpoint-dir', help='按冻结计划/环境校验并恢复完整日结果；不覆盖已完成单位')
    for command in (freeze, run):
        command.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        raise FileExistsError('Output directory must not exist')
    if args.command == 'freeze':
        result = freeze_dynamic(args.config, args.library)
        report = '# 冻结真实多周OE动态计划\n\n尚未计算后段收益。\n\n```json\n' + pretty_json(result) + '\n```\n'
        publish_bundle(result, report, args.output_dir, 'plan.json')
    else:
        result = run_dynamic(args.plan, detail=args.detail, checkpoint_dir=args.checkpoint_dir)
        publish_bundle(result, render_dynamic(result), args.output_dir, 'result.json')
    print(f'已保存 {args.command} 产物：{args.output_dir}')
    return result


if __name__ == '__main__':
    main()
