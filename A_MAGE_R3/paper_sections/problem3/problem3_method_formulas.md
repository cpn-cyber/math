# 第三问方法公式汇总

## 1. 当前复评公式

$$
Q_{{cur}}=\\alpha F_1+(1-\\alpha)F_2,\\quad \\alpha=0.80.
$$

其中 $F_1$ 来自问题1封版综合评价模型，$F_2$ 来自问题2关键特征辅助模型。

## 2. 五元论证链

$$
T\\rightarrow D\\rightarrow HM\\rightarrow R\\rightarrow C\\rightarrow T.
$$

$$
\\Gamma_i=\\sum_e \\omega_e s_e
=0.20s_{{TD}}+0.25s_{{DHM}}+0.25s_{{HMR}}+0.20s_{{RC}}+0.10s_{{CT}}.
$$

$$
G_i=1-\\Gamma_i.
$$

## 3. D-S证据融合

$$
m_k(A)=\\rho_k e_k,\\quad m_k(H)=\\rho_k(1-e_k),\\quad m_k(U)=1-\\rho_k.
$$

Dempster组合规则为

$$
m_{12}(X)=\\frac{\\sum_{B\\cap C=X}m_1(B)m_2(C)}{1-K},\\quad
K=\\sum_{B\\cap C=\\varnothing}m_1(B)m_2(C).
$$

融合后以

$$
R_{{AI}}=BetP(A)
$$

作为AI辅助写作风险提示指标。该指标仅表示文本存在 AI辅助写作风险，不构成学术不端判断，也不等同于AI生成判定。

## 4. 多智能体评分与分歧

$$
\\overline Q_i=\\frac1R\\sum_{r=1}^{R}Q_i^{{(r)}},\\quad
D_i=std(Q_i^{{(1)}},\\ldots,Q_i^{{(R)}}).
$$

$$
LCB(Q_i)=\\overline Q_i-z_\\gamma D_i.
$$

## 5. 修改动作向量与优化

$$
a_j=(\\Delta Q_j,c_j,\\Delta G_j,\\Delta R_{{AI,j}},risk_j).
$$

$$
\\max \\sum_j x_j\\Delta Q_j,\quad
\\sum_j x_jc_j\\le B,\quad x_j\\in\\{{0,1\\}}.
$$

同时关注 $G_i$、$R_{{AI}}$ 和多智能体分歧 $D_i$ 的下降。

## 6. 优化后预测

概念模型：

$$
Q_{{new}}=\\alpha F_{{1,new}}+(1-\\alpha)F_{{2,new}}-\\lambda_GG_{{new}}-\\lambda_AR_{{AI,new}}.
$$

Step29封版实现：

$$
score\\_gain_i=\\min(0.12E_i,12),
$$

$$
\\hat Q_i=\\min(100,Q_{{cur,i}}+score\\_gain_i).
$$

AI辅助风险和评审分歧预测为

$$
R_{{AI,after}}=\\max(0,R_{{AI,before}}-\\Delta R_{{AI}}),
$$

$$
D_{after}=\\max(0,D_{before}-\\Delta D).
$$

## 7. 稳健性检验

Step30扰动参数包括收益映射系数、修改预算、风险下降倍率和分歧收敛倍率。对参数组合 $\\theta$，记录

$$
\\hat Q_i(\\theta),\\quad R_{{AI,i}}(\\theta),\\quad D_i(\\theta),\\quad x_j(\\theta).
$$

动作稳定性可写为

$$
P(a_j)=\\frac{\\#\\{{\\theta:a_j\\text{{ 被选中}}\\}}}{\\#\\{{\\theta\\}}}.
$$
