# Little-Go：已验证的优化提交版

2026-10-14新增Stage2候选：`stage2_candidate/my_player3.py`，压缩包 `little-go-stage2-candidate.zip`。只建议在Stage2单独验证；Stage1保留当前已拿到75/75的版本。完整选择和验收证据见 `reports/STRONG-CANDIDATE.md`。此候选尚无官方Stage2成绩，不能把本地强对手对照当作Championship胜率。

## 使用与提交

提交文件为 `submission/my_player3.py`。只把这个文件上传到 Vocareum 的 work 目录；Stage 1 和 Stage 2 可以使用同一文件。不要上传 reference、tests、tools、reports 或实验模型。

选手使用标准库，在当前目录读取 `input.txt`，生成 `output.txt`。它会自动创建可选的 `state.json` 保存回合记录，不需要提前上传该文件。开始新局会重置记录；记录损坏或与输入不匹配时不信任其回合数，仍从输入棋盘判断合法落子。

代码默认每步搜索预算最多 3 秒 CPU，并加上墙钟截止时间；实际部署还需要在 Vocareum 验证。这是时间安全设置，不是胜率保证。提交版采用位棋盘、缓存合法后继、迭代加深 PVS / alpha-beta、合法动作排序和带界限类型的置换表。评估保持原有存活棋子、气和威胁权重；KO 仍按当前历史单独核对。

本地调用示例（先准备符合 PDF 格式的 input.txt）：

```text
python3 my_player3.py
```

`LITTLE_GO_CPU_SECONDS` 环境变量仅用于减少开发搜索时间，不允许把预算提高到 3 秒以上。正式提交无需设置。

## 已验证内容与结果

最新优化结果见 `reports/OPTIMIZATION.md`、`reports/STAGE1-pvs-resume-heldout.md` 和 `reports/stage1-pvs-resume-heldout-summary.json`。旧 `summary.json`、`RESULTS.md`、`STAGE1.md` 保留的是原基线结果。原始比赛和败局棋谱保存在各 JSON 中，不能混淆不同版本、预算和种子。

本轮完整默认预算对战：保留测试集 Random 19/20、Greedy 代理 20/20、Aggressive 代理 20/20、Alpha-beta 代理 20/20；另一个同种子开发集 Alpha-beta 从旧版 16/20 提升到 20/20。四组均达到公开的 18/20 阈值，本地 Stage1 公式折算 90/90，不能等同于正式得分。全部 100 局、2,400 步经课程 GO 独立重放，无非法落子、超时或崩溃。

- 规则使用课程裁判和独立矩阵实现交叉核对。
- 测试覆盖正交连接、去重气、多群提子、捕获型自杀例外、KO、PASS、终局 komi、输出格式、跨进程回合记录、损坏记录、超时备用落子及置换表复用。
- 已实际使用 CPython 3.7.5 运行测试，不只检查新 Python 的语法兼容。
- 完整文件对战验证使用未经修改的 `host.py`，并每回合重新启动进程。
- 本地 Greedy / Aggressive / Alpha-beta 是原创代理，并非评分服务器中的选手。
- 未提供隐藏评分选手和 Vocareum 计时包装器，故本地结果不能直接等同于成绩。

## 规则边界

PDF 规定两次连续 PASS 结束。提供的文件 host 会在黑方初始空棋盘直接 PASS 时立即结束；选手黑方开局有合法放置时不会 PASS。

白方收到两张空棋盘，在 PDF 规则下表示黑方开局 PASS；选手识别该状态。普通相同前后棋盘表示对手 PASS。

提供的 `GO.play()` 未正确实现两次连续 PASS，不能直接作为可靠的测试比赛循环。本地 harness 实现明确的 PDF 和 official-file 模式，最终文件入口另有整局验证。

第 24 步先执行行动和提子，再结算。评分只统计最终留在棋盘的棋子，白方加 2.5；不把空地、眼或累计提子数当作分数。

## 开发测试

Stage 1 完整本地演练使用 `tools/stage1.py`，需要 Unix 环境，以真实子进程运行提交文件并核算子进程 CPU。每个对手运行 20 局，黑白各 10 局，默认 3 秒预算，不设置开发降时环境变量。四组报告完成后由 `tools/stage1_summary.py` 汇总。该演练仍不等同于 Vocareum 的正式选手测试。

```text
python3 tools/stage1.py --opponent official-random --games 20 --seed 561100
python3 tools/stage1.py --opponent greedy-proxy --games 20 --seed 561100
python3 tools/stage1.py --opponent aggressive-proxy --games 20 --seed 561100
python3 tools/stage1.py --opponent alphabeta-proxy --games 20 --seed 561100
python3 tools/stage1_summary.py
```

在 `little-go-hw2` 目录运行：

```text
python3 -m unittest discover -s tests -v
python3 tools/check_official.py --states 1000
python3 tools/benchmark.py --opponent official-random --games 200 --cpu 0.025 --output reports/random-baseline.json
python3 tools/benchmark.py --opponent alphabeta-proxy --games 40 --cpu 0.75 --seed 30000 --output reports/alphabeta-075.json
python3 tools/probe_deployment.py --games 2
```

比赛工具记录种子、代码指纹、CPU 与墙钟耗时、逐色结果、非法/超时/崩溃计数和棋谱。按时间截止的搜索在不同机器及负载下可能完成不同深度；固定种子不意味着整个时间受限代理会逐步完全相同。

## 学习实验

`tools/train_value.py` 为本地原创线性 TD(0) 位置价值实验，保持 Black 视角并在 White 评估时翻转符号。八种空间对称的棋盘和动作变换有测试覆盖；这里不进行黑白互换。

```text
python3 tools/train_value.py --games 1000
python3 tools/compare_learning.py --games 40 --cpu 0.025
```

本轮学习与混合评估劣于纯搜索，模型只存于 reports，不参与提交。训练胜负不作为测试胜率。仅在后续保留集和可访问正式对手的结果证明改进时考虑部署学习模型。

## 当前局限

这是有规则、接口和本地比赛证据的原创优化版。可访问的四类 Stage1 测试已全部达到 18/20 阈值；正式 Alpha-beta 和 Stage2 的 Q-learning、Championship 仍需要官方报告。尚无正式 100/100 成绩证据。

没有从网上或其他学生取得 Little-Go 解法。reference 中保留的是用户提供、课程允许参考的源码副本；部署搜索、规则和学习实验独立编写。课程关于 AI 辅助的规定尚未提供，不据此推断允许范围。
