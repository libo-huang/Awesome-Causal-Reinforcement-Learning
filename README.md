# Awesome Causal Reinforcement Learning

[![](https://awesome.re/badge.svg)](#awesome-causal-reinforcement-learning)
[![](https://img.shields.io/badge/Contributions-Welcome-1f425f)](#clap-contributions-chinese-version)
[![](https://img.shields.io/static/v1?label=%E2%AD%90&message=If%20Useful&style=flat&color=C7A5C0)](https://github.com/libo-huang/Awesome-Causal-Reinforcement-Learning)
[![](https://img.shields.io/github/last-commit/libo-huang/Awesome-Causal-Reinforcement-Learning.svg)](#awesome-causal-reinforcement-learning)

## :pushpin: Contents

[:information_desk_person: Abstract](#information_desk_person-abstract)

[:closed_book: Surveys](#closed_book-surveys)

[:bookmark_tabs: Papers](#bookmark_tabs-papers)

&emsp; [2024](#2024) | [2023](#2023) | [2022](#2022) | [2021](#2021) | [2020](#2020) | [2019](#2019) | [2018](#2018) | [2017](#2017) | [Pre-2017](#pre-2017)

[:clap: Contributions](#clap-contributions-chinese-version)

---




















## :information_desk_person: Abstract

Causal Reinforcement Learning (CRL) is a suite of algorithms, embedding causal knowledge into RL for more efficient and effective model learning, policy evaluation, or policy optimization.
How causality information inspires current RL algorithms is illustrated in the below CRL framework,

<div align="center">
  <img src="./utils/sketch_map_CRL.png" alt="" width="500">
</div>

> CRL framework illustrates how causality information inspires current RL algorithms. This framework contains possible algorithmic connections between planning and causality-inspired learning procedures. Explanations of each arrow are,
>
> - a) input training data for the causal representation or abstraction learning;
> - b) input representations, abstractions, or training data from the real world for the causal model;
> - c) plan over a learned or given causal model,
> - d) use information from a policy or value network to improve the planning procedure,
> - e) use the result from planning as training targets for a policy or value,
> - g) output an action in the real world from the planning,
> - h) output an action in the real world from the policy/value function,
> - f) input causal representations, abstractions, or training data from the real world for the policy or value update.

Note that most CRL algorithms implement only a subset of the possible connections with causality, enjoying potential benefits in data efficiency, interpretability, robustness, or generalization of the model or policy.

We detailed a comprehensive survey of CRL in the paper, [**A survey on causal reinforcement learning**](https://ieeexplore.ieee.org/abstract/document/10771589), and particularly list the causal-reinforcement-related works in this repository.

‼️ We have been continuously updating the latest papers, so the scope of literature covered extends beyond the above survey.

⁉️ Any new related works are welcome to be added via [pull requests](#clap-contributions-chinese-version).

If you find the paper useful, please cite with,

```bibtex
@article{zeng2024survey,
  title={A survey on causal reinforcement learning},
  author={Zeng, Yan and Cai, Ruichu and Sun, Fuchun and Huang, Libo and Hao, Zhifeng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```

<div align="right">
  <a href="#awesome-causal-reinforcement-learning" style="font-size: 40px;">:top:</a>
</div>

---




















## :closed_book: Surveys
- (**TNNLS 2024**) A survey on causal reinforcement learning [[paper](https://ieeexplore.ieee.org/abstract/document/10771589)]
- (**TNNLS 2023**) A survey on reinforcement learning for recommender systems [[paper](https://ieeexplore.ieee.org/abstract/document/10144689)]
- (**arxiv 2022**) Causal machine learning: A survey and open problems [[paper](https://arxiv.org/pdf/2206.15475)]
- (**NeurIPS-W 2021**) Causal multi-agent reinforcement learning: Review and open problems [[paper](https://arxiv.org/pdf/2111.06721)]
- (**ICML Tutorials 2020**) Causal reinforcement learning [[tutorial](https://crl.causalai.net/)]
- (**Blog 2018**) Introduction to causal reinforcement learning [[blog](https://causallu.com/2018/12/31/introduction-to-causalrl/)]
- (自动化学报 **2024**) 基于因果建模的强化学习控制: 现状及展望 [[论文](http://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c220823)]

<!--- (**ACM Computing Surveys 2017**) Imitation learning: A survey of learning methods [[paper](https://www.open-access.bcu.ac.uk/5045/1/Imitation%20Learning%20A%20Survey%20of%20Learning%20Methods.pdf)]
- (**IJRR 2013**) Reinforcement learning in robotics: A survey [[paper](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Kober_IJRR_2013.pdf)] -->

<!--- (**ACM Computing Surveys 2021**) Reinforcement learning in healthcare: A survey [[paper](https://arxiv.org/pdf/1908.08796)] [[code](https://dl.acm.org/doi/abs/10.1145/3477600)] 
%- (**Frontiers in genetics 2019**) Review of Causal Discovery Methods Based on Graphical Models [[paper](https://par.nsf.gov/servlets/purl/10125762)]-->

<div align="right">
  <a href="#awesome-causal-reinforcement-learning" style="font-size: 40px;">:top:</a>
</div>

---





















## :bookmark_tabs: Papers
### 2024



- (**ICML 2024**) Policy learning for balancing short-term and long-term rewards [[paper](https://openreview.net/pdf?id=7Qf1uHTahP)] [[code](https://github.com/YanaZeng/Short_long_term-Rewards)]
- (**ICML 2024**) ACE: Off-Policy Actor-Critic with Causality-Aware Entropy Regularization [[paper](https://proceedings.mlr.press/v235/ji24b.html)] [[code](https://github.com/jity16/ACE-Off-Policy-Actor-Critic-with-Causality-Aware-Entropy-Regularization)]
- (**ICML 2024**) Causal Action Influence Aware Counterfactual Data Augmentation [[paper](https://proceedings.mlr.press/v235/armengol-urpi-24a.html)] [[code](https://github.com/martius-lab/caiac)]
- (**ICML 2024**) Learning Causal Dynamics Models in Object-Oriented Environments [[paper](https://proceedings.mlr.press/v235/yu24j.html)] [[code](https://github.com/EaseOnway/oocdm)]
- (**ICML 2024**) Agent-Specific Effects: A Causal Effect Propagation Analysis in Multi-Agent MDPs [[paper](https://proceedings.mlr.press/v235/triantafyllou24a.html)] [[code](https://github.com/stelios30/agent-specific-effects)]
- (**ICML 2024**) Tackling Non-Stationarity in Reinforcement Learning via Causal-Origin Representation [[paper](https://proceedings.mlr.press/v235/zhang24ah.html)] [[code](https://github.com/PKU-RL/COREP)]
- (**ICML 2024**) Fine-Grained Causal Dynamics Learning with Quantization for Improving Robustness in Reinforcement Learning [[paper](https://proceedings.mlr.press/v235/hwang24b.html)] [[code](https://github.com/iwhwang/Fine-Grained-Causal-RL)]
- (**ICML 2024**) Causal Bandits: The Pareto Optimal Frontier of Adaptivity, a Reduction to Linear Bandits, and Limitations around Unknown Marginals [[paper](https://proceedings.mlr.press/v235/liu24b.html)]
- (**AAAI 2024**) ACAMDA: Improving Data Efficiency in Reinforcement Learning Through Guided Counterfactual Data Augmentation [[paper](https://openreview.net/pdf?id=4pjgPGB1qr)] 
- (**IJCAI 2024**) Boosting Efficiency in Task-Agnostic Exploration through Causal Knowledge [[paper](https://arxiv.org/pdf/2407.20506)] [[code](https://github.com/CMACH508/CausalExploration)]
- (**JASA 2024**) Off-policy confidence interval estimation with confounded Markov decision process [[paper](https://arxiv.org/pdf/2202.10589)] [[code](https://github.com/callmespring/cope)]







### 2023

- (**NeurIPS 2023**) Learning world models with identifiable factorization [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/65496a4902252d301cdf219339bfbf9e-Paper-Conference.pdf)] [[code](https://github.com/AlexLiuyuren/IFactor)]
- (**NeurIPS 2023**) Interpretable reward redistribution in reinforcement learning: a causal approach [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/402e12102d6ec3ea3df40ce1b23d423a-Paper-Conference.pdf)]
- (**ICLR 2023**) Causal Confusion and Reward Misidentification in Preference-Based Reward Learning [[paper](https://arxiv.org/pdf/2204.06601)]
- (**TPAMI 2023**) Invariant policy learning: A causal perspective [[paper](https://ieeexplore.ieee.org/abstract/document/10005169)]
- (**TNNLS 2023**) Sample efficient deep reinforcement learning with online state abstraction and causal transformer model prediction [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10219179)]
- (**TII 2023**) Spatial-temporal causality modeling for industrial processes with a knowledge-data guided reinforcement learning [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10352645)]
- (**JRSSB 2023**) Estimating and improving dynamic treatment regimes with a time-varying instrumental variable [[paper](https://arxiv.org/pdf/2104.07822)]
- (**Operations Research 2023**) Proximal Reinforcement Learning: Efficient Off-Policy Evaluation in Partially Observed Markov Decision Processes [[paper](https://pubsonline.informs.org/doi/abs/10.1287/opre.2021.0781)] [[code](https://github.com/CausalML/ProximalRL)]
- (**The Annals of Statistics 2023**) Off-policy evaluation in partially observed Markov decision processes [[paper](https://arxiv.org/pdf/2110.12343)]
- (**arxiv 2023**) MACCA: Offline Multi-agent Reinforcement Learning with Causal Credit Assignment [[paper](https://openreview.net/pdf?id=mFBR2ksIwY)]







### 2022

- (**JMLR 2022**) On instrumental variable regression for deep offline policy evaluation [[paper](https://www.jmlr.org/papers/volume23/21-0614/21-0614.pdf)] [[code](https://github.com/liyuan9988/IVOPEwithACME)]
- (**TNNLS 2022**) Fully decentralized multiagent communication via causal inference [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9761961)]
- (**ICLR 2022**) On covariate shift of latent confounders in imitation and reinforcement learning [[paper](https://openreview.net/pdf?id=w01vBAcewNX)] [[code](https://openreview.net/attachment?id=w01vBAcewNX&name=supplementary_material)]
- (**ICLR 2022**) A relational intervention approach for unsupervised dynamics generalization in model-based reinforcement learning [[paper](https://openreview.net/pdf?id=YRq0ZUnzKoZ)] [[code](https://github.com/CR-Gjx/RIA)]
- (**ICLR 2022**) Causal contextual bandits with targeted interventions [[paper](https://openreview.net/pdf?id=F5Em8ASCosV)]  [[code](https://openreview.net/forum?id=F5Em8ASCosV)]
- (**ICLR 2022**) Adarl: What, where, and how to adapt in transfer reinforcement learning [[paper](https://openreview.net/pdf?id=8H5bpVwvt5)] [[code](https://github.com/Adaptive-RL/AdaRL-code)]
- (**NeurIPS 2022**) Generalizing goal-conditioned reinforcement learning with variational causal reasoning [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a96368eb38bce0956a1132154d70d72d-Paper-Conference.pdf)] [[code](https://github.com/GilgameshD/GRADER)]
- (**NeurIPS 2022**) Causality-driven hierarchical structure discovery for reinforcement learning [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/7e9fbd01b3084956dd8a070c7bf30bad-Paper-Conference.pdf)]
- (**NeurIPS 2022**) Factored Adaptation for Non-stationary Reinforcement Learning [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/cf4356f994917177213c55ff438ddf71-Paper-Conference.pdf)]
- (**NeurIPS 2022**) Online reinforcement learning for mixed policy scopes [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/15349e1c554406b7719d047a498e7117-Abstract-Conference.html)]
- (**NeurIPS 2022**) Sequence Model Imitation Learning with Unobserved Contexts [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/708e58b0b99e3e62d42022b4564bad7a-Paper-Conference.pdf)] [[code](https://github.com/gkswamy98/sequence_model_il)]
- (**ICML 2022**) Fighting fire with fire: avoiding dnn shortcuts through priming [[paper](https://proceedings.mlr.press/v162/wen22d/wen22d.pdf)] [[code](https://github.com/AlvinWen428/fighting-fire-with-fire)]
- (**ICML 2022**) A minimax learning approach to off-policy evaluation in confounded partially observable Markov decision processes [[paper](https://proceedings.mlr.press/v162/shi22f/shi22f.pdf)] [[code](https://github.com/jiaweihhuang/Confounded-POMDP-Exp)]
- (**ICML 2022**) Action-sufficient state representation learning for control with structural constraints [[paper](https://proceedings.mlr.press/v162/huang22f/huang22f.pdf)]
- (**ICML 2022**) Causal dynamics learning for task-independent state abstraction [[paper](https://proceedings.mlr.press/v162/wang22ae/wang22ae.pdf)] [[code](https://github.com/wangzizhao/CausalDynamicsLearning)]
- (**ICML 2022**) Causal imitation learning under temporally correlated noise [[paper](https://proceedings.mlr.press/v162/swamy22a/swamy22a.pdf)] [[code](https://github.com/gkswamy98/causal_il)]
- (**ECCV 2022**) Resolving copycat problems in visual imitation learning via residual action prediction [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990386.pdf)] [[code](https://github.com/linYDTHU/Resolving_copycat_problems_via_residual_prediction)]
- (**AAAI 2022**) Invariant action effect model for reinforcement learning [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20913)]
- (**CHIL 2022**) Counterfactually Guided Policy Transfer in Clinical Settings [[paper](https://proceedings.mlr.press/v174/killian22a/killian22a.pdf)]
- (**CLeaR 2022**) Efficient Reinforcement Learning with Prior Causal Knowledge [[paper](https://proceedings.mlr.press/v177/lu22a/lu22a.pdf)]
- (**CHIL 2022**) Counterfactually Guided Policy Transfer in Clinical Settings [[paper](https://proceedings.mlr.press/v174/killian22a/killian22a.pdf)]
- (**ICLR-W 2022**) Invariant causal representation learning for generalization in imitation and reinforcement learning [[paper](https://openreview.net/forum?id=r87xPSd89gq)]
- (**arXiv 2022**) Offline reinforcement learning with causal structured world models [[paper](https://arxiv.org/pdf/2206.01474)]






### 2021

- (**TNNLS 2021**) Model-based transfer reinforcement learning based on graphical model representations [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9540989)]
- (**ICML 2021**) Keyframe-focused visual imitation learning [[paper](https://proceedings.mlr.press/v139/wen21d.html)] [[code](https://github.com/AlvinWen428/keyframe-focused-imitation-learning)]
- (**ICML 2021**) Causal curiosity: Rl agents discovering self-supervised experiments for causal representation learning [[paper](https://proceedings.mlr.press/v139/sontakke21a.html)] [[code](https://github.com/sumedh7/CausalCuriosity)]
- (**ICML 2021**) Model-free and model-based policy evaluation when causality is uncertain [[paper](https://brunssmith.com/wp-content/uploads/2021/06/confound_mdp_icml.pdf)]
- (**ICML 2021**) A spectral approach to off-policy evaluation for pomdps [[paper](https://lyang36.github.io/icml2021_rltheory/camera_ready/61.pdf)]
- (**ICLR 2021**) Learning ”what-if” explanations for sequential decision-making [[paper](https://arxiv.org/pdf/2007.13531)]
- (**ICLR 2021**) Learning invariant representations for reinforcement learning without reconstruction [[paper](https://arxiv.org/pdf/2006.10742)] [[code](https://github.com/facebookresearch/deep_bisim4control)]
- (**NeurIPS 2021**) Invariant causal imitation learning for generalizable policies [[paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/204904e461002b28511d5880e1c36a0f-Abstract.html)]
- (**NeurIPS 2021**) Provably efficient causal reinforcement learning with confounded observational data [[paper](https://proceedings.neurips.cc/paper/2021/file/b0b79da57b95837f14be95aaa4d54cf8-Paper.pdf)]
- (**NeurIPS 2021**) Causal Influence Detection for Improving Efficiency in Reinforcement Learning [[paper](https://proceedings.neurips.cc/paper/2021/hash/c1722a7941d61aad6e651a35b65a9c3e-Abstract.html)] [[code](https://github.com/martius-lab/cid-in-rl)]
- (**NeurIPS 2021**) Deep proxy causal learning and its application to confounded bandit policy evaluation [[paper](https://proceedings.neurips.cc/paper/2021/file/dcf3219715a7c9cd9286f19db46f2384-Paper.pdf)] [[code](https://github.com/liyuan9988/DeepFeatureProxyVariable/)]
- (**NeurIPS 2021**) Sequential causal imitation learning with unobserved confounders [[paper](https://proceedings.neurips.cc/paper/2021/file/7b670d553471ad0fd7491c75bad587ff-Paper.pdf)]
- (**NeurIPS 2021**) Causal bandits with unknown graph structure [[paper](https://proceedings.neurips.cc/paper/2021/hash/d010396ca8abf6ead8cacc2c2f2f26c7-Abstract.html)]
- (**IJCAI 2021**) Inferring time-delayed causal relations in pomdps from the principle of independence of cause and mechanism [[paper](https://par.nsf.gov/servlets/purl/10279293)]
- (**AAAI 2021**) Reinforcement learning of causal variables using mediation analysis [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20648)] [[code](https://gitlab.compute.dtu.dk/tuhe/causal_nie)]
- (**ICRA 2021**) Causal reasoning in simulation for structure and transfer learning of robot manipulation policies [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9561439)]
- (**WWW 2021**) Cost-effective and interpretable job skill recommendation with deep reinforcement learning [[paper](https://dl.acm.org/doi/pdf/10.1145/3442381.3449985)] [[code](https://github.com/sunyinggilly/SkillRec)]
- (**WWW 2021**) Unifying Offline Causal Inference and Online Bandit Learning for Data Driven Decision [[paper](https://dl.acm.org/doi/abs/10.1145/3442381.3449982)]
- (**AISTATS 2021**) Budgeted and non-budgeted causal bandits [[paper](https://proceedings.mlr.press/v130/nair21a/nair21a.pdf)]
- (**AISTATS 2021**) Off-policy evaluation in infinite-horizon reinforcement learning with latent confounders [[paper](https://proceedings.mlr.press/v130/bennett21a/bennett21a.pdf)]
- (**CORL 2021**) SMARTS: An Open-Source Scalable Multi-Agent RL Training School for Autonomous Driving [[paper](https://proceedings.mlr.press/v155/zhou21a/zhou21a.pdf)] [[code](https://github.com/huawei-noah/SMARTS)]
- (**UAI 2021**) Bandits with partially observable confounded data [[paper](https://proceedings.mlr.press/v161/tennenholtz21a/tennenholtz21a.pdf)]
- (**MICAI 2021**) Causal based action selection policy for reinforcement learning [[paper](https://www.researchgate.net/profile/Arquimides-Mendez-Molina/publication/355566090_Causal_Based_Action_Selection_Policy_for_Reinforcement_Learning/links/63fc43c20cf1030a5655ac7f/Causal-Based-Action-Selection-Policy-for-Reinforcement-Learning.pdf)]
- (**Management Science 2021**) Minimax-optimal policy learning under unobserved confounding [[paper](https://par.nsf.gov/servlets/purl/10168529)]
- (**Proceedings of the IEEE 2021**) Toward causal representation learning [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9363924)]
- (**INAOE report 2021**) Combining reinforcement learning and causal models for robotics application [[report](http://imav2020.inaoep.mx/archivos/CCC-20-005.pdf)]
- (**L4DC 2021**) Invariant Policy Optimization: Towards Stronger Generalization in Reinforcement Learning [[paper](https://proceedings.mlr.press/v144/sonar21a/sonar21a.pdf)] [[code](https://github.com/irom-lab/Invariant-Policy-Optimization)]
- (**arXiv 2021**) Causal reinforcement learning using observational and interventional data [[paper](https://arxiv.org/pdf/2106.14421)] [[code](https://github.com/causal-rl-anonymous/causal-rl)]
- (**ICLR-W 2021**) Resolving causal confusion in reinforcement learning via robust exploration [[paper](https://openreview.net/pdf?id=DKCXncD4Xtq)]
- (**ICLR-W 2021**) Modelinvariant state abstractions for model-based reinforcement learning [[paper](https://arxiv.org/pdf/2102.09850)]
- (**arxiv 2021**) Causal reinforcement learning: An instrumental variable approach [[paper](https://arxiv.org/pdf/2103.04021)]
- (**arxiv 2021**) Instrumental variable value iteration for causal offline reinforcement learning [[paper](https://arxiv.org/pdf/2102.09907)]
- (**arXiv 2021**) Causaldyna: Improving generalization of dyna-style reinforcement learning via counterfactual-based data augmentation [[paper](https://openreview.net/pdf/feb5d2e66742510f4a27ef832c2b3d59ea3ef68d.pdf)]
- (**arXiv 2021**) Causal imitative model for autonomous driving [[paper](https://arxiv.org/pdf/2112.03908)] [[code](https://github.com/vita-epfl/CIM)]






### 2020

- (**ICML 2020**) Invariant causal prediction for block mdps [[paper](https://proceedings.mlr.press/v119/zhang20t/zhang20t.pdf)] [[code](https://github.com/facebookresearch/icp-block-mdp)]
- (**ICLR 2020**) Causalworld: A robotic manipulation benchmark for causal structure and transfer learning [[paper](https://openreview.net/forum?id=SK7A5pdrgov)] [[code](https://github.com/rr-learning/CausalWorld)]
- (**ICML 2020**) Designing optimal dynamic treatment regimes: A causal reinforcement learning approach [[paper](https://proceedings.mlr.press/v119/zhang20a/zhang20a.pdf)]
- (**NeurIPS 2020**) Fighting copycat agents in behavioral cloning from observation histories [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/1b113258af3968aaf3969ca67e744ff8-Paper.pdf)] [[code](https://github.com/AlvinWen428/fighting-copycat-agents)]
- (**NeurIPS 2020**) Causal imitation learning with unobserved confounders [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/8fdd149fcaa7058caccc9c4ad5b0d89a-Paper.pdf)]
- (**NeurIPS 2020**) Off-policy policy evaluation for sequential decisions under unobserved confounding [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/da21bae82c02d1e2b8168d57cd3fbab7-Paper.pdf)] [[code](https://github.com/StanfordAI4HI/off_policy_confounding)]
- (**AAAI 2020**) Off-Policy Evaluation in Partially Observable Environments [[paper](https://cdn.aaai.org/ojs/6590/6590-13-9815-1-10-20200520.pdf)]
- (**AAAI 2020**) Causal transfer for imitation learning and decision making under sensor-shift [[paper](https://cdn.aaai.org/ojs/6571/6571-13-9796-1-10-20200519.pdf)]
- (**Master thesis 2020**) Structural Causal Models for Reinforcement Learning [[thesis](https://escholarship.mcgill.ca/downloads/x920g245x)]
- (**UAI 2020**) Regret Analysis of Bandit Problems with Causal Background Knowledge [[paper](https://arxiv.org/pdf/1910.04938)]
- (**IROS 2020**) Learning transition models with time-delayed causal relations [[paper](https://par.nsf.gov/servlets/purl/10191470)]
- (**ICLR-W 2020**) Resolving spurious correlations in causal models of environments via interventions [[paper](https://arxiv.org/pdf/2002.05217)]
- (**NeurIPS-W 2020**) Sample-efficient reinforcement learning via counterfactual-based data augmentation [[paper](https://offline-rl-neurips.github.io/pdf/34.pdf)]
- (**arXiv 2020**) Causally correct partial models for reinforcement learning [[paper](https://arxiv.org/pdf/2002.02836)]
- (**arXiv 2020**) Causality and batch reinforcement learning: Complementary approaches to planning in unknown domains [[paper](https://arxiv.org/pdf/2006.02579)]









### 2019

- (**Nature 2019**) Grandmaster level in StarCraft II using multi-agent reinforcement learning [[paper](https://www.nature.com/articles/s41586-019-1724-z)]
- (**Science 2019**) Human-level performance in 3d multiplayer games with population-based reinforcement learning [[paper](https://people.eecs.berkeley.edu/~russell/classes/cs294/f21/papers/Jaderberg-etal-2019-Science-Capture-The-Flag.pdf)] [[code](https://www.science.org/doi/10.1126/science.aau6249)]
- (**NeurIPS 2019**) Causal confusion in imitation learning [[paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/947018640bf36a2bb609d3557a285329-Paper.pdf)] [[code](https://github.com/pimdh/causal-confusion)]
- (**NeurIPS 2019**) Policy evaluation with latent confounders via optimal balance [[paper](https://papers.nips.cc/paper_files/paper/2019/file/7c4bf50b715509a963ce81b168ca674b-Paper.pdf)] [[code](https://github.com/CausalML/LatentConfounderBalancing)]
- (**NeurIPS 2019**) Near-optimal reinforcement learning in dynamic treatment regimes [[paper](https://papers.nips.cc/paper_files/paper/2019/file/8252831b9fce7a49421e622c14ce0f65-Paper.pdf)]
- (**ICML 2019**) Counterfactual off-policy evaluation with gumbel-max structural causal models [[paper](https://proceedings.mlr.press/v97/oberst19a/oberst19a.pdf)] [[code](https://github.com/clinicalml/gumbel-max-scm)]
- (**ICML 2019**) Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning [[paper](https://proceedings.mlr.press/v97/jaques19a/jaques19a.pdf)]
- (**ICLR 2019**) Woulda, coulda, shoulda: Counterfactually-guided policy search [[paper](https://openreview.net/pdf?id=BJG0voC9YQ)]
- (**AAAI 2019**) Virtual-taobao: Virtualizing real-world online retail environment for reinforcement learning [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/4419)] [[code](https://github.com/eyounx/VirtualTaobao)]
- (**AAAI 2019**) Structural causal bandits with non-manipulable variables [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/4320)]
- (**ICCV 2019**) Exploring the limitations of behavior cloning for autonomous driving [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Codevilla_Exploring_the_Limitations_of_Behavior_Cloning_for_Autonomous_Driving_ICCV_2019_paper.pdf)] [[code](https://github.com/felipecode/coiltraine/blob/master/docs/exploring_limitations.md)]
- (**KDD 2019**) Environment reconstruction with hidden confounders for reinforcement learning based recommendation [[paper](https://d1wqtxts1xzle7.cloudfront.net/115305373/1907-libre.pdf?1716685533=&response-content-disposition=inline%3B+filename%3DEnvironment_Reconstruction_with_Hidden_C.pdf&Expires=1723185179&Signature=DNjD-KkhrlKrlJUUjQZRYh8nzARIKOV71VxAvFq5dJzhJBolNwVG7hQf616BJDllYLdn1QZ22M1KKvl0zX21TH9gzfWqkkx3ELL5tiBmsa0LqPUJL2vfzE8nYo5UcwlPsnIsS1Q3dBMv6Lhj7sGbv93x-sXROLjZ4269LdIXJsjb9Z7bwkKBT~79nQiwf1MJgMSrSoyb9xrWqiRoLOufYsIOBBrQa-EffP39LYAllLyO3l9HHcTKI6PYHNQ7-JQ3YPa3sO0ZnODBUrpMja9Qhq1vRiTMFC~3GF-fhkjP9AnHGSsek6~tXHQWVZC2mmuZYTQOyqYFVvRL5PODmCjpWQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)]
- (**arXiv 2019**) Causal reasoning from meta-reinforcement learning [[paper](https://arxiv.org/pdf/1901.08162)]
- (**arXiv 2019**) Learning causal state representations of partially observable environments [[paper](https://arxiv.org/pdf/1906.10437)]
- (**arXiv 2019**) Causal Induction from Visual Observations for Goal Directed Tasks [[paper](https://arxiv.org/pdf/1910.01751)] [[code](https://github.com/StanfordVL/causal_induction)]





### 2018

- (**MIT press 2018**) Reinforcement learning: An introduction [[book](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)]
- (**Basic books 2018**) The Book of Why: the new science of cause and effect [[book](http://repo.darmajaya.ac.id/5342/1/The%20book%20of%20why_%20the%20new%20science%20of%20cause%20and%20effect%20%28%20PDFDrive%20%29.pdf)]
- (**ICML 2018**) Causal Bandits with Propagating Inference [[paper](https://proceedings.mlr.press/v80/yabe18a/yabe18a.pdf)]
- (**NeurIPS 2018**) Structural Causal Bandits: Where to Intervene? [[paper](https://papers.nips.cc/paper_files/paper/2018/file/c0a271bc0ecb776a094786474322cb82-Paper.pdf)] [[code](https://github.com/sanghack81/SCMMAB-NIPS2018)]
- (**NeurIPS 2018**) Confounding-robust policy improvement [[paper](https://papers.nips.cc/paper_files/paper/2018/file/3a09a524440d44d7f19870070a5ad42f-Paper.pdf)] [[code](https://github.com/CausalML/confounding-robust-policy-improvement)]
- (**AAAI 2018**) Learning plannable representations with causal infogan [[paper](https://proceedings.neurips.cc/paper/2018/hash/08aac6ac98e59e523995c161e57875f5-Abstract.html)] [[code](https://github.com/thanard/causal-infogan)]
- (**AAAI 2018**) Counterfactual multi-agent policy gradients [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11794)] [[code](https://github.com/matteokarldonati/Counterfactual-Multi-Agent-Policy-Gradients)]
- (**AAAI 2018**) Deep reinforcement learning that matters [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11694)] [[code](https://github.com/Breakend/DeepReinforcementLearningThatMatters)]
- (**IHSED 2018**) Measuring collaborative emergent behavior in multi-agent reinforcement learning [[paper](https://arxiv.org/pdf/1807.08663)]
- (**Foundations and Trends in Robotics 2018**) An algorithmic perspective on imitation learning [[paper](https://rl.aalto.fi/publication/2018algorithmic/2018algorithmic.pdf)]
- (**ICML-W 2018**) Playing against nature: causal discovery for decision making under uncertainty [[paper](https://arxiv.org/pdf/1807.01268)]
- (**arXiv 2018**) Deconfounding reinforcement learning in observational settings [[paper](https://github.com/CausalRL/DRL)] [[code](https://github.com/CausalRL/DRL)]




### 2017

- (**MIT press 2017**) Elements of causal inference: foundations and learning algorithms [[book](https://library.oapen.org/bitstream/handle/20.500.12657/26040/11283.pdf?sequence=1&isAllowed=y)]
- (**PhD thesis 2017**) Cognitive robotic imitation learning system based on cause-effect reasoning [[thesis](https://api.drum.lib.umd.edu/server/api/core/bitstreams/252d6555-63e5-457e-bd84-fbc94ee21af0/content)]
- (**TCDS 2017**) A novel parsimonious cause-effect reasoning algorithm for robot imitation and plan recognition [[paper](https://ieeexplore.ieee.org/abstract/document/7812655)] [[code](https://github.com/garrettkatz/ceril)]
- (**ICML 2017**) Neural Episodic Control [[paper](https://proceedings.mlr.press/v70/pritzel17a.html?ref=https://githubhelp.com)]
- (**ICML 2017**) Schema networks: Zero-shot transfer with a generative causal model of intuitive physics [[paper](https://proceedings.mlr.press/v70/kansky17a.html)]
- (**ICML 2017**) Counterfactual Data-Fusion for Online Reinforcement Learners [[paper](https://proceedings.mlr.press/v70/forney17a/forney17a.pdf)]
- (**ICML 2017**) Identifying Best Interventions through Online Importance Sampling [[paper](https://proceedings.mlr.press/v70/sen17a/sen17a.pdf)]
- (**IJCAI 2017**) Transfer learning in multi-armed bandit: a causal approach [[paper](https://www.ijcai.org/proceedings/2017/0186.pdf)]
- (**TACON 2017**) Infinite time horizon maximum causal entropy inverse reinforcement learning [[paper](https://ieeexplore.ieee.org/abstract/document/8115277)]
- (**CoRL 2017**) CARLA: An Open Urban Driving Simulator [[paper](https://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf)] [[code](https://github.com/carla-simulator/carla)]




### Pre-2017

- (**NIPS 2016**) Causal bandits: Learning good interventions via causal inference [[paper](https://proceedings.neurips.cc/paper/2016/file/b4288d9c0ec0a1841b3b3728321e7088-Paper.pdf)] [[code](https://github.com/finnhacks42/causal_bandits)]
- (**ICAGI 2016**) Imitation learning as cause-effect reasoning [[paper](https://www.academia.edu/download/101188481/katz_et_al_AGI2016.pdf)] [[code](https://github.com/garrettkatz/copct)]
- (**Technical report 2016**) Markov decision processes with unobserved confounders: A causal approach [[paper](https://www.cs.purdue.edu/homes/eb/mdp-causal.pdf)]
- (**NIPS 2015**) Bandits with unobserved confounders: A causal approach [[paper](https://proceedings.neurips.cc/paper/2015/file/795c7a7a5ec6b460ec00c5841019b9e9-Paper.pdf)] [[code](https://github.com/ucla-csl/mabuc)]

<!--
- (**ICML 2016**) Guided cost learning: Deep inverse optimal control via policy optimization [[paper](https://proceedings.mlr.press/v48/finn16.pdf)]
- (**Applied informatics 2016**) Causal discovery and inference: concepts and recent methodological advances [[paper](https://www.researchgate.net/profile/Peter-Spirtes/publication/295088352_Causal_discovery_and_inference_concepts_and_recent_methodological_advances/links/570f95a908aec95f0614da48/Causal-discovery-and-inference-concepts-and-recent-methodological-advances.pdf)]
- (**Nature 2015**) Human-level control through deep reinforcement learning [[paper](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)]
- (**Cambridge University Press 2015**) Causal inference in statistics, social, and biomedical sciences [[book](https://wiki.swarma.org/images/5/54/-Guido_W._Imbens%2C_Donald_B._Rubin-_Causal_Inferenc%28z-lib.org%29.pdf)]
- (**TIT 2013**) The principle of maximum causal entropy for estimating interacting processes [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6479340)]
- (**NIPS 2013**) Playing atari with deep reinforcement learning [[paper](https://arxiv.org/pdf/1312.5602)]
- (**ICML 2010**) Modeling interaction via the principle of maximum causal entropy [[paper](https://icml.cc/Conferences/2010/papers/28.pdf)]
- (**ICML 2004**) Apprenticeship learning via inverse reinforcement learning [[paper](https://icml.cc/Conferences/2004/proceedings/papers/335.pdf)]
- (**MIT press 2001**) Causation, Prediction, and Search [[book](https://philarchive.org/archive/SPICPA-2)]
- (**Cambridge University Press 2000**) Causality: Models, Reasoning, and Inference [[book](https://archive.org/details/causalitymodelsr0000pear/page/n9/mode/2up)]
-->

<div align="right">
  <a href="#awesome-causal-reinforcement-learning" style="font-size: 40px;">:top:</a>
</div>

---















## :clap: Contributions [[chinese version](https://blog.csdn.net/HLBoy_happy/article/details/140026155?fromshare=blogdetail&sharetype=blogdetail&sharerId=140026155&sharerefer=PC&sharesource=HLBoy_happy&sharefrom=from_link)]

**1. Fork the Repository:** Click on the `Fork` button in the top-right corner to create a copy of the repository in your GitHub account.

**2. Create a New Branch:** In your forked repository, create a new branch (e.g., "libo") by using the branch selector button near the top-left (usually labeled `master` or `main`).

**3. Make Your Changes:** Switch to your new branch using the same selector. Then, click the `Edit file` button at the top right and make your changes. Add entries in the following format:

```bash
  - (**publisher_name year**) manuscript_name [[publication_type](online_manuscript_link)] [[code](online_code_link)]
```

**4. Commit Changes:** Save your changes by clicking the `Commit changes` button in the upper-right corner. Enter a commit message (e.g., "add 1 cvpr'24 paper") and an extended description if necessary, then confirm your changes by clicking the `Commit changes` button again at the bottom right.

**5. Create a Pull Request:** Go back to your forked repository and click `Compare & pull request`. Alternatively, select your branch from the branch selector and click `Open pull request` from the `Contribute` drop-down menu. Fill out the title and description for your pull request, and click `Create pull request` to submit it.


<div align="right">
  <a href="#awesome-causal-reinforcement-learning" style="font-size: 40px;">:top:</a>
</div>
<div align="center">
  <a href="#awesome-causal-reinforcement-learning">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=libo-huang.awesome-causal-reinforcement-learning&left_color=blue&right_color=red&format=true" alt="Visitor Badge">
  </a>
</div>
