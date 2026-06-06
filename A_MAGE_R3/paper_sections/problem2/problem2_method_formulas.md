# 第二问方法公式整理

## 1. 稳健标准化
$$z_{ij}=\frac{x_{ij}-\operatorname{median}(x_j)}{\operatorname{MAD}(x_j)+\varepsilon},\quad \operatorname{MAD}(x_j)=\operatorname{median}(|x_{ij}-\operatorname{median}(x_j)|).$$

## 2. Spearman秩相关
$$r_s=1-\frac{6\sum_{i=1}^{n}d_i^2}{n(n^2-1)},$$
其中 $d_i$ 为两个变量秩次之差。本文样本量为10，因此主要用于关系强弱描述，不强调显著性。

## 3. 灰色关联度
$$\xi_j(i)=\frac{\Delta_{\min}+\rho\Delta_{\max}}{|x_0(i)-x_j(i)|+\rho\Delta_{\max}},\quad G_j=\frac{1}{n}\sum_{i=1}^{n}\xi_j(i).$$
本文取分辨系数 $\rho=0.5$。

## 4. 初步关键性指数 K_pre
$$K_{pre,j}=0.6|r^S_j|+0.4G_j.$$

## 5. PLS模型
PLS通过提取潜变量同时解释特征矩阵 $X$ 和质量标签 $Q$，本文候选潜变量数为 $A\in\{1,2\}$，并通过LOOCV选择。由于PLS输入使用Step14全样本稳健标准化后的特征矩阵，该结果应表述为近似留一审计，用于VIP解释和小样本稳定性诊断，不作为严格外部泛化验证。

## 6. VIP指标
$$VIP_j=\sqrt{p\frac{\sum_{a=1}^{A}SS_a w_{ja}^2/\|w_a\|^2}{\sum_{a=1}^{A}SS_a}},$$
其中 $SS_a$ 表示第 $a$ 个潜变量对因变量的解释贡献。

## 7. QAF质量调整因子
$$u_i=\operatorname{mean}(task\_coverage_i,data\_credibility_i,method\_fit_i,formula\_explanation_i,result\_closure_i)-stacking\_penalty_i,$$
$$u_i^c=u_i-\overline{u},$$
$$\phi_i=\operatorname{clip}(1+\eta u_i^c,0.90,1.10),$$
$$\hat Q^{QAF}_i=\operatorname{clip}(\phi_i\tilde Q_i,Q_{\min},Q_{\max}).$$

## 8. Bootstrap稳定性
每次从10篇论文中有放回抽取10篇，重新拟合PLS(A=1)，记录VIP、VIP>1概率和系数符号一致率。

## 9. 最终关键特征指数 K_final
$$K_j=0.35|r^S_j|+0.25G_j+0.25VIP^{norm}_j+0.15SC_j.$$

## 10. 成对排序加权差分
$$pair\_score_{ij}=\sum_{k\in\mathcal{F}}K_k(x_{ik}-x_{jk}),$$
若 $pair\_score_{ij}>0$，则预测论文 $i$ 优于论文 $j$；反之预测论文 $j$ 优于论文 $i$。该检验的真实方向由弱监督标签 $Q_i-Q_j$ 诱导，因此只作为关键特征对封版系统排序的一致性辅助检查，不作为独立外部验证。

