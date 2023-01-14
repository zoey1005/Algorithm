$$Objective(\phi ) = E(x,y) \sim D_{\pi _{\phi }^{RL}}[r_{\theta }(x,y)-\beta log(\pi ^\frac{RL}{\phi}(y|x)/\pi^{SFT}(y|x))]+\gamma E_{x}\sim D_{pretain}[log(\pi ^\frac{RL}{\phi }(x))]$$
​
>目标=得分-差异（以人为主）+泛化能力
SFT有监督模型
y|x：把一句话x输入进去后，有监督模型会输出一个y
$\pi ^\frac{RL}{\phi}(y|x)/\pi^{SFT}(y|x)$: 强化学习和有监督模型分析差异的比较
以人为主：有监督学习模型放在分母
泛化能力：下游任务（多目标），如情感分析、机器翻译等等

​
