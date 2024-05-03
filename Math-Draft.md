$\bm{y}和\bm{w}的概率分布:$
$$
p(\bm{y}, \bm{w}|\bm{X}) = p(\bm{y}\bm{w},\bm{X}) p(\bm{w}) = p(\bm{w})\prod_{i=1}^{N}p(y_i|\bm{w},\bm{x}_i) \\
$$


$根据贝叶斯公式，模型参数\bm{w}的后验分布表示为：$
$$
p(\bm{w}|\bm{X},\bm{y})=\frac{p(\bm{y}, \bm{w}|\bm{X})p(\bm{w})}{p(\bm{y}|\bm{X})} \propto p(\bm{w}) \prod_{i=1}^{N}p(y_i|\bm{w}, \bm{x}_i)
$$
$\\$
$设参数\bm{w}的先验分布是偏移参数为\mu=0，尺度参数为b的拉普拉斯分布，概率分布：$
$$p(\bm{w}|\mu=0,b)=\frac{1}{2b}\exp(-\frac{|\bm{w}|}{b})$$
$即\bm{w} 服从拉普拉斯分布，表示为 \bm{w} \sim \text{Laplace}(0, b)$


$假设y_i和\bm{x}_i和\bm{w}之间的关系服从高斯分布，满足：$
$$
\begin{align*}
y_i &\sim \mathcal{N}(\bm{x}_i^T \bm{w}, \sigma^2) \\
p(y_i | \bm{x}_i, \bm{w}, \sigma^2) &\sim \mathcal{N}(y_i | \bm{x}_i^T \bm{w}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp(-\frac{(y_i - \bm{w}^T\bm{x}_i)^2}{2\sigma^2})
\end{align*}
$$


$那么后验分布可以表示为：$
$$
\begin{align*}
p(\bm{w}|\bm{X},\bm{y}) & \propto p(\bm{w}) \prod_{i=1}^{N}p(y_i|\bm{w}, \bm{x}_i) 
\\
& \propto p(\bm{w} | \mu=0,b) \prod_{i=1}^{N} p(y_i | \bm{x}_i, \bm{w}, \sigma^2)
\\
& = p(\bm{w} | \mu=0,b) \prod_{i=1}^{N} \mathcal{N}(y_i | \bm{x}_i^T \bm{w}, \sigma^2)
\end{align*}
$$

$最大后验估计MAP：$
$$
\begin{align*}
\arg \max_{\bm{w}} p(\bm{w}|\bm{X}, \bm{y}) 
& = \arg \max_{\bm{w}} \left( p(\bm{w} | \mu=0,b) \prod_{i=1}^{N} \mathcal{N}(y_i | \bm{x}_i^T \bm{w}, \sigma^2)\right)
\\
& = \arg \max_{\bm{w}} \left(\frac{1}{2b}\exp(-\frac{|\bm{w}|}{b}) \prod_{i=1}^{N}\frac{1}{\sqrt{2\pi\sigma^2}} \exp(-\frac{(y_i - \bm{w}^T\bm{x}_i)^2}{2\sigma^2})\right)
\\
&\text{取对数，连乘变连加}
\\
& = \arg\max_{\bm{w}}\log \left(\frac{1}{2b}\exp(-\frac{|\bm{w}|}{b}) \prod_{i=1}^{N}\frac{1}{\sqrt{2\pi\sigma^2}} \exp(-\frac{(y_i - \bm{w}^T\bm{x}_i)^2}{2\sigma^2})\right)
\\
& = \arg\max_{\bm{w}} \left(\log(\frac{1}{2b}\exp(-\frac{|\bm{w}|}{b})) \sum_{i=1}^{N}\log(\frac{1}{\sqrt{2\pi\sigma^2}} \exp(-\frac{(y_i - \bm{w}^T\bm{x}_i)^2}{2\sigma^2}))\right)
\\
& \log(\frac{1}{2b})\text{和}\log(\frac{1}{\sqrt{2\pi\sigma^2}})\text{是常数，对}\bm{w}\text{没影响，可以忽略}
\\
& = \arg\max_{\bm{w}}\left(-\frac{|\bm{w}|}{b} - \frac{1}{2\sigma^2} \sum_{i=1}^{N}(y_i-\bm{w}^T\bm{x}_i)^2 \right)
\\
& = \arg \min_{\bm{w}}\left(\frac{|\bm{w}|}{b} + \frac{1}{2\sigma^2} \sum_{i=1}^{N}(y_i-\bm{w}^T\bm{x}_i)^2 \right)
\end{align*}
$$

因此，最终要解决的优化问题是：
$$
\min_{\bm{w}}\left(\frac{|\bm{w}|}{b} + \frac{1}{2\sigma^2} \sum_{i=1}^{N}(y_i-\bm{w}^T\bm{x}_i)^2 \right)
$$

这个优化目标函数，相当于给线性模型添加了L1和L2正则化约束：
- $\frac{|\bm{w}|}{b}$是关于$\bm{w}$的L1正则化
- $ \frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i - \bm{w}^T\bm{x}_i)^2 $ 是平方误差损失项，为 L2 正则化项,可以看作是 $ w $ 的 L2 范数的平方乘以系数 $ \frac{1}{2\sigma^2} $