"""冻结并运行真实多周OE成熟覆盖/专家资格/模型版本就绪审计。"""
import argparse
from pathlib import Path

from src.artifacts import pretty_json
from src.multiweek_readiness import freeze_readiness, render_readiness, run_readiness
from src.session_experiment import publish_bundle


def main(argv=None):
    """数据/库使用既有入口生成；此处只接受冻结计划，另存JSON与中文报告。"""
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    freeze=commands.add_parser('freeze');freeze.add_argument('--config',required=True);freeze.add_argument('--library',required=True)
    run=commands.add_parser('run');run.add_argument('--plan',required=True);run.add_argument('--detail',action='store_true')
    for c in (freeze,run):c.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    if Path(args.output_dir).exists():raise FileExistsError('Output directory must not exist')
    if args.command=='freeze':
        result=freeze_readiness(args.config,args.library)
        report='# 冻结真实多周就绪计划\n\n尚不计算收益。\n\n```json\n'+pretty_json(result)+'\n```\n'
        publish_bundle(result,report,args.output_dir,'plan.json')
    else:
        result=run_readiness(args.plan,detail=args.detail)
        publish_bundle(result,render_readiness(result),args.output_dir,'result.json')
    print(f'已保存 {args.command} 产物：{args.output_dir}')
    return result


if __name__=='__main__':main()
