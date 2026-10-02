"""冻结并运行最近/平移成熟历史窗口 OE-ARS 开发双对照。"""
import argparse
from pathlib import Path

from src.ars_experiment import freeze_ars_experiment,render_ars_experiment,run_ars_experiment
from src.artifacts import pretty_json
from src.session_experiment import publish_bundle


def main(argv=None):
    """只读取冻结参数；输出目录必须新建，不能覆盖旧窗口与正式结果。"""
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    freeze=commands.add_parser('freeze');freeze.add_argument('--config',required=True);freeze.add_argument('--library',required=True)
    run=commands.add_parser('run');run.add_argument('--plan',required=True);run.add_argument('--detail',action='store_true')
    for command in (freeze,run):command.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    if Path(args.output_dir).exists():raise FileExistsError('Output directory must not exist')
    if args.command=='freeze':
        result=freeze_ars_experiment(args.config,args.library)
        report='# 冻结历史窗口 ARS 双对照计划\n\n尚不计算收益。\n\n```json\n'+pretty_json(result)+'\n```\n'
        publish_bundle(result,report,args.output_dir,'plan.json')
    else:
        result=run_ars_experiment(args.plan,detail=args.detail)
        publish_bundle(result,render_ars_experiment(result),args.output_dir,'result.json')
    print(f'已保存 {args.command} 产物：{args.output_dir}')
    return result


if __name__=='__main__':main()
