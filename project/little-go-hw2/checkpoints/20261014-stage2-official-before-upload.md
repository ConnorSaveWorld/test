---
title: Little-Go official Stage2 test before upload
timestamp: 2026-10-14
branch: none (not a Git repository)
status: in-progress
---

# 续聊记录：官方 Stage2 测试尚未开始

用户最新要求：保存聊天记录，token不足。停止当前浏览器操作，保存上下文；不要继续上传或提交。旁边 recent-chat-summaries.json 保存了工具可取得的最近30轮聊天摘要，不是完整逐字聊天导出。

## 用户目标与授权

用户要完成 USC CSCI561 HW2 Little-Go，尽量拿100分。一直要求中文、直接负责实现、先测试再提交，避免让用户反复耗时跑平台。已授权助手在已登录Vocareum运行官方Stage2测试；不需要重新问“要不要测”。但任何满分保证均无证据。新候选已完成本地开发及验证，下一步是官方测试，不再重跑相同代理测试替代正式对手。

工作目录 C:/Users/paul2/.codex/.chatgpt-projects/g-p-68dc33922ddc819197d2694abbc00426/little-go-hw2。不是Git仓库；sources/和原课程下载均只读，不要初始化Git或操作无关WSL仓库。

## 确认的平台记录

- 当前课程标题20253-CSCI-561 (Foundations of AI)，HW2，Stage2工作区。
- 课程页面显示总分80/100，Stage1显示75/90、graded submission number2。之前公开测试报告75/75：Random20/20，Greedy20/20，Aggressive18/20。
- Stage2曾Q9/10、Championship0/10，Final score5。最新报告TueOct13 12:09:36PDT：Q7/10（黑2/5白5/5），Champ0/10（黑白各0/5），Final score3.5。
- Stage2当前提交次数3，最近提交Oct13 11:56:25amPDT。页面记分与最新测试报告不完全一致；分别记录，不假定最终成绩规则。
- 用户提供Downloads/my_player3.py，经SHA核对是原PVS dda83...，不是新连接性候选。平台当前文件也已实际下载验证为同一指纹。
- 隐藏Alpha-beta15分、候选官方Q/Champ成绩仍未知。

## 两份代码，避免混淆

1. Stage1原版：submission/my_player3.py、little-go-submission.zip；SHA256 dda83b2ac22036ea4bdc49418193aa38893fe4f1f6c64d06494e2a45156bff36。保留Stage1原提交。
2. Stage2候选：stage2_candidate/my_player3.py、little-go-stage2-candidate.zip；SHA256 4a219c507c529f466235eeebeded1467fdb4323496963068fceef370b264eb9f。22132字节，ZIP5827字节且只有my_player3.py，字节一致。等同experiments/candidate_connect2.py。

候选只加零和的连接群−2/+2评估项，规则、计分、KO、PVS和默认3秒预算保持原样。眼评估、5秒预算、战术叶扩展都因本地退步被排除。详情reports/STRONG-CANDIDATE.md。

## 已做完的本地验证

- 独立强搜索代理配对开局：开发旧36/64、新42/64；另一个独立seed630000、同0.25秒目标旧33/64、新41/64。不是官方Champ仿真，开局强制且样本相关，不转换为Champ胜率。
- 默认3秒实际CPython3.7.5、每步新进程：四组各20/20，总80/80，全部1920步课程GO重放通过，无非法/超时/崩溃；CPU总1880.066秒，maxmove3.278秒，maxgame25.976秒。
- 候选27项规则/协议/搜索测试通过，内部25000次双oracle规则核对零差异；全开发60项测试通过。两个原文件host接口探针双方均赢，非胜率估计。
- 独立只读review_strong_final审查通过，额外120位置/30固定深度验证。工具来源指纹问题已修正为源字节快照，两个文件变更回归通过。当前完整80局是在旧加载工具启动后、源文件冻结状态下完成的，未冒充新快照保护。
- 用户提供文件中没有官方Q-learning/Championship本地可运行对手；只有Random/manualGo和TicTacToe starter。

## 浏览器当前状态与操作记录

用户已亲自登录。使用mcp__cua_repl，浏览器IAB id2，tab1。持久JS句柄vocareum（后续同session可复用；新session先cua.getState/getTab并读新文档）。原课程URL https://labs.vocareum.com/main/main.php?m=course&cid=226300 。

当前Stage2工作区URL https://labs.vocareum.com/main/main.php?m=editor&mode=s&asnid=5714859&stepid=5714861 。已再次markHandoff，保留登录页签，但仍需检查会话是否有效。

目前只做了查看和备份，**未上传新候选、未点击Submit、未启动任何新官方测试**。

已在Details -> View Submission Report实际看到最新3.5报告。当前work/my_player3.py是旧PVS。点Download先弹“Preparing ZIP File”模态，需点击Download zipped source链接才能真正下载。成功下载后用本地zip核对只有work/my_player3.py，SHA dda83...。备份已复制到 backups/vocareum-stage2-before-test-20261014.zip（不要覆盖）。

备份下载原路径：C:/Users/paul2/Downloads/yanxiabu@usc.edu_5363616_s5363616_5714861_Oct_14_2026_12-17-25pm_PDT.zip。

最新截图：工作区窄屏524px，顶栏Submit/Details/Grades/Actions；左侧tree有resource（lib/asnlib/scripts/startercode，锁定）、work/my_player3.py、Submissions（锁定）。工具栏只显示Rename/Delete/Copy/Download，没有明显Upload。Actions只有Download startercode、Applications、Reset assignment。对work按钮right-click没有展开上传菜单。不能盲点重置或删除。

来源编辑器已加载旧代码，但窄屏源代码区截图看起来空白；AX/DOM仍有Source code editor及代码行，可能布局原因。可以放大浏览器视口后找文件New/Upload控件，或通过编辑器现有正常UI粘贴候选并保存。先观察真实DOM/截图，不调用页面私有JS/API，不用外部HTTP模拟浏览器。编辑器paste前必须先从local候选文件取得完整内容，之后下载并核对SHA再Submit。

浏览器文件上传文档已读：先waitForEvent('filechooser')，再点实际input[type=file]或关联可见上传按钮，用chooser.setFiles(绝对文件路径)。没有locator.setInputFiles。此前只找了菜单，没有触发filechooser。

Terminal iframe处于Launch Terminal按钮，同时有bash提示；未执行任何远程命令。不要记录/传播iframe中带token的URL。

## 后续顺序

1. 恢复本记录，确认用户愿意继续当前测试（“继续”已足够，不重复权限问题），检查现存已登录tab。
2. 只在Stage2的work根目录上传/正常UI编辑新候选my_player3.py；Stage1不改。备份已完成。重新下载Stage2文件并核对4a219c...。
3. 点击Stage2 Submit；这是用户已授权的测试动作。若页面出现新的法律条款/CAPTCHA等特殊门槛，按浏览器确认政策处理，不能猜测绕过。
4. 验证界面显示新的提交号/队列或运行状态，截图证明；等待官方报告。用户估计耗时约1小时，不静默轮询，不让用户重复提交。必要时用线程heartbeat继续跟踪（只有用户明确要求监测时创建）；每次等待/阻塞不得长于60秒。
5. 记录两对手逐色胜率、正式分数、代码指纹、提交号。新候选若未改善，不虚称满分，保留原版备份。

没有当前长期计算进程；所有本地测试worker已完成。两个只读review subagent均完成。无goal工具创建的活动目标，勿凭空标完成。
