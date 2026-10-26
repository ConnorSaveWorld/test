"""Summarize completed local Stage1 rehearsal; never predicts official grades."""
import json
import argparse
from benchmark import ROOT


def points(name,w):
    if w>=.9:
        return 15 if name=='alphabeta-proxy' else 25
    if name in ('official-random','greedy-proxy'):
        return 15*w if w>=.7 else 0
    if name=='aggressive-proxy':
        return (20 if w>=.7 else 10)*w
    return (10 if w>=.7 else 5)*w


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    prefix='stage1-'+(args.tag+'-' if args.tag else '')
    names=('official-random','greedy-proxy','aggressive-proxy','alphabeta-proxy')
    reports=[json.loads((ROOT/'reports'/(prefix+name+'.json')).read_text()) for name in names]
    if any(r['games']!=20 or len(r['results'])!=20 for r in reports):
        raise RuntimeError('Require completed 20-game reports for all four opponents')
    total=sum(r['total_agent_cpu'] for r in reports)
    failed={kind:sum(r['failures'][kind] for r in reports) for kind in ('invalid','timeout','crash')}
    summary={'kind':'local fresh-process Stage1 rehearsal, not Vocareum grading',
             'games':80,'total_agent_cpu':total,'stage_cpu_limit':7200,
             'within_local_stage_cpu_limit':total<=7200,'failures':failed,
             'source_sha256':reports[0]['source_sha256'],'opponents':[]}
    lines=['# Stage 1 本地完整演练','',
           '实际 CPython 3.7.5，直接运行提交文件，每回合重新启动；默认预算最多 3 秒 CPU，使用真实跨回合辅助记录。',
           '', '| 对手 | 黑方胜/局 | 白方胜/局 | 总胜/局 | 达到 18/20 | 累计选手 CPU | 最长单步 CPU | 本地公式折算 |',
           '|---|---:|---:|---:|---|---:|---:|---:|']
    for r in reports:
        if r['source_sha256']!=summary['source_sha256']:
            raise RuntimeError('Different submission versions in rehearsal')
        b,w=r['by_color']['1'],r['by_color']['2']
        row={k:r[k] for k in ('opponent','games','wins','by_color','mean_cpu','max_cpu','max_wall','max_game_cpu','total_agent_cpu','wilson95','failures')}
        row['meets_18_of_20']=r['wins']>=18
        row['proxy_rubric_points']=points(r['opponent'],r['win_rate'])
        summary['opponents'].append(row)
        lines.append('| {} | {}/{} | {}/{} | {}/{} | {} | {:.2f} 秒 | {:.3f} 秒 | {:.2f} |'.format(r['opponent'],b['wins'],b['games'],w['wins'],w['games'],r['wins'],r['games'],'是' if row['meets_18_of_20'] else '否',r['total_agent_cpu'],r['max_cpu'],row['proxy_rubric_points']))
    summary['local_proxy_points']=sum(r['proxy_rubric_points'] for r in summary['opponents'])
    lines+=['','80 局总选手 CPU：{:.2f} 秒；本地核算与 7200 秒阶段上限比较：{}。'.format(total,'通过' if total<=7200 else '超出'),
            '非法行动：{}；CPU 超时/进程截止超时：{}；崩溃：{}。'.format(failed['invalid'],failed['timeout'],failed['crash']),
            '', '按公开公式对本地对手结果折算：{:.2f}/90。这是本地演练指标，不是正式成绩或预测。'.format(summary['local_proxy_points']),
            '', 'Random 使用课程原选手；其余三个是本地代理，并非 Vocareum 评分选手。Alpha-beta 正式选手仍为隐藏对手。本轮 20 局样本不证明稳定总体胜率；时间受限的搜索也可能受机器负载影响。',
            '', '原始逐局结果、种子、行动、搜索深度、CPU 和墙钟记录保存在 stage1-<opponent>.json。各工作进程并行运行，各自使用 RUSAGE_CHILDREN 统计自身选手子进程 CPU，没有把四组耗时按墙钟混算。']
    summary_name=prefix+'summary.json'
    document_name='STAGE1'+('-'+args.tag if args.tag else '')+'.md'
    (ROOT/'reports'/summary_name).write_text(json.dumps(summary,indent=2))
    (ROOT/'reports'/document_name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
