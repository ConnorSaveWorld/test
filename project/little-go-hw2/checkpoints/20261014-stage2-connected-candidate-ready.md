# Stage2 independent candidate ready — 2026-10-14

User provided all available aggregate reports, no detailed game logs. Original PVS gotStage1 75/75, Stage2 first5points (Q9/10 Champ0/10), latest3.5points (Q7/10 Champ0/10). The latest user-provided Downloads/my_player3.py SHA dda83... is exactly original PVS, not the newly tested candidate. Do not overwrite Downloads or assume new candidate has already failedofficialChampionship.

## Artifacts

- Original Stage1 delivery remains submission/my_player3.py and little-go-submission.zip, SHA dda83b2ac22036ea4bdc49418193aa38893fe4f1f6c64d06494e2a45156bff36.
- New Stage2-only candidate: stage2_candidate/my_player3.py (22132bytes), little-go-stage2-candidate.zip (5827bytes, onlymy_player3.py, byte-identical).
- CandidateSHA4a219c507c529f466235eeebeded1467fdb4323496963068fceef370b264eb9f, identicaltoexperiments/candidate_connect2.py.
- Current report reports/STRONG-CANDIDATE.md; platform report reports/VOCAREUM-STAGE2-20261013-1209.md.

## Empirical selection and validation

Only evalgroup featurechanged: −2 perBlackgroup +2 perWhitegroup, signedforcurrentplayer; rules, terminalstone+komi, PVS, KO, helper, default3sec unchanged. Eyes,5sec-budgetandquiescence experiments regressedandrejected. Developmentlocalpool42/64vsold36; separate630000fixtures/equalCPUtarget0.25 41/64vsold33. Thesearecorrelatedforcedopeningfixturesvsoriginalfulllegal-searchprofiles, notofficialChampionshiprates. Replayofbothverified.

Fullfresh-process3.7.5 regressions80games: Random20/20, GreedyProxy20/20, AggressiveProxy20/20, AlphaProxy20/20; all1920actionsreplayedwithoriginalGO,zeroerrors. TotalCPU1880.066,maxmoveCPU3.278,maxgameCPU25.976. Twoactualfilehostgamesdefaultbudgetpassedbothcolors. Candidate25000rulecomparisonszero mismatches; actualexport27rule/protocol/search testsallpassed; entiredevelopment60testspassed. Independentread-onlyreviewfoundnocandidateblockers,120ruleand30searchfixturechecks passed.

Review reporting issuefixed: leaguecompilesexactbytesandretainscandidate/opponenthash; Stage1runsprivatebytesnapshot. Mutationtestsfailedthenpassed,includingactualchildexecution. Already-running80jobsloadedpriorharnessandusedunchangedfrozencandidate; donotretroactivelyclaimnewsnapshotexecution.

## Next action

Deliver newStage2candidate and instruct uploadonlyitsmy_player3.py intoStage2workroot, Submitthere. PreserveStage1prior75/75. FormalnewcandidateQ/ChampresultsandhiddenAlpha15remainunknown. No100/100guarantee and no automaticVocareumsubmissionperformed.

Alllocaltournamentsandprobescompleted. Revieweragent /root/review_strong_final completed bothreviews. NoGitinitializationorcommits; noeditsunderreadonlysources/originalcoursefiles. RuntimecontinuesWSL Ubuntu-24.04 .runtime/Python-3.7.5/python.exe.
