# 用户提供的 Vocareum Stage2 测试结果

报告执行时间：Tue Oct 13 09:15:13 PDT 2026。用户粘贴平台输出；未直接访问平台提交记录，输出未包含源文件指纹、逐步棋谱或 CPU 明细。

| 官方对手 | 黑方胜/局 | 白方胜/局 | 总胜/局 | 胜率 |
|---|---:|---:|---:|---:|
| qlearning_player | 4/5 | 5/5 | 9/10 | 0.9 |
| championship_player | 0/5 | 0/5 | 0/10 | 0 |

平台原文评分：Final score: 5；Mission Completed。Q-learning 达到满分阈值并取得5分；Championship未获得分数。Stage2可见结果为5/10。加上先前Stage1公开75/75，已测部分合计80/85；隐藏Alpha-beta15分仍未知。当前结果不是总成绩100/100。

## 调查边界

这份汇总说明程序能被平台识别并赢下9局Q-learning，但不提供Championship每局的终局原因、棋盘、动作、用时或辅助记录连续性。因此无法据此区分正常棋力差距、某类非法落子、计时问题或跨回合记录失效。没有证据时不将任何原因写成定论，也不直接覆盖已拿到公开80分的源文件。

优先取得至少一局黑方和一局白方的完整Championship棋谱、结束原因及计时。使用原课程GO重放，核对每个输入/输出、捕获、KO、PASS和终局计分；然后定位首次导致不可逆败局的决策。新增候选须独立保存，并重新验证Stage1和Q-learning没有明显退步。单纯提高搜索时间或替换为未经验证的学习模型不视作已解决问题。

## 用户报告核心原文

```text
Programming language...
PY3

Tue 13 Oct 2026 09:02:03 AM PDT

Playing with qlearning_player...
Tue 13 Oct 2026 09:02:03 AM PDT
=====Summary=====
You play as Black Player | Win: 4 | Lose: 1 | Tie: 0
You play as White Player | Win: 5 | Lose: 0 | Tie: 0

Playing with championship_player...
Tue 13 Oct 2026 09:07:22 AM PDT
=====Summary=====
You play as Black Player | Win: 0 | Lose: 5 | Tie: 0
You play as White Player | Win: 0 | Lose: 5 | Tie: 0

Result:
qlearning win rate: 0.9
championship win rate: 0
Final score: 5

Mission Completed.
Tue 13 Oct 2026 09:15:11 AM PDT
```
