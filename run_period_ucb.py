"""冻结并运行固定期间 OE-UCB 等权奖励对照，保留全部年龄与静态基线。"""
import argparse
from pathlib import Path

from src.artifacts import pretty_json
from src.period_experiment import freeze_period_experiment, render_period_experiment, run_period_experiment
from src.session_experiment import publish_bundle


def main(argv=None):
    """模型库先另存建库；run 只接收冻结计划，不覆盖参数或现有产物。"""
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    freeze=commands.add_parser('freeze');freeze.add_argument('--config',required=True);freeze.add_argument('--library',required=True)
    run=commands.add_parser('run');run.add_argument('--plan',required=True);run.add_argument('--detail',action='store_true')
    for command in (freeze,run):command.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    if Path(args.output_dir).exists():raise FileExistsError('Output directory must not exist')
    if args.command=='freeze':
        result=freeze_period_experiment(args.config,args.library)
        report='# 冻结固定期间 OE-UCB 计划\n\n尚不计算收益。\n\n```json\n'+pretty_json(result)+'\n```\n'
        publish_bundle(result,report,args.output_dir,'plan.json')
    else:
        result=run_period_experiment(args.plan,detail=args.detail)
        publish_bundle(result,render_period_experiment(result),args.output_dir,'result.json')
    print(f'已保存 {args.command} 产物：{args.output_dir}')
    return result


if __name__=='__main__':main()
