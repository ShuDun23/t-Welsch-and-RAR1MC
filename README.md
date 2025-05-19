# t-Welsch & RAR1MC
Code for ___Robust Rank-One Matrix Completion via Explicit Regularizer___ accepted by TNNLS 2025
> **Abstract**
> <p> &nbsp; In robust matrix completion (MC), Welsch function, also referred to as the maximum correntropy criterion with Gaussian kernel, has been widely employed. However, it suffers from the drawback of down-weighing normal data. This work is the first to uncover the explicit regularizer (ER) for the Welsch function based on the multiplicative form of half-quadratic minimization. Leveraging this discovery, we develop a new function called \emph{t-Welsch}, also with ER, which provides unity weight to normal data and exhibits stronger robustness against large-magnitude outliers compared to Huber's weight. We apply the t-Welsch to rank-one matching pursuit, enabling accurate and robust low-rank matrix recovery without the need of rank information and singular value decomposition. The resultant MC algorithm is realized via block coordinate descent, whose analyses of convergence and computational complexity are produced. Experiments are conducted using synthetic random data, as well as real-world images with salt-and-pepper noise and multiple-input multiple-output radar signals in the presence of Gaussian mixture disturbances. In all three scenarios, the proposed algorithm outperforms the state-of-the-art robust MC methods in terms of recovery accuracy. <p>

All experiments are conducted using MATLAB r2024b.

- To reproduce Figure 8, please run main1.m

- To reproduce Table III & IV, please run main2.m

- In addition, we provide two detailed examples reproducing Fig.5(b) and Fig.6(c).

For detailed hyperparameter settings please refer to the latest paper.

If you have any question, please feel free to reach out hnsheng2-c@my.cityu.edu.hk
