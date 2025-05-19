# t-Welsch & RAR1MC
Code for ___Robust Rank-One Matrix Completion via Explicit Regularizer___ accepted by TNNLS 2025
> **Abstract:**  
> In robust matrix completion (MC), Welsch function, also referred to as the maximum correntropy criterion with Gaussian kernel, has been widely employed. However, it suffers from the drawback of down-weighing normal data. This work is the first to uncover the ___explicit regularizer___ (ER) for the Welsch function based on the multiplicative form of half-quadratic minimization. Leveraging this discovery, we develop a new function called ___t-Welsch___, also with ER, which provides unity weight to normal data and exhibits stronger robustness against large-magnitude outliers compared to Huber's weight. We apply the t-Welsch to rank-one matching pursuit, enabling accurate and robust low-rank matrix recovery without the need of rank information and singular value decomposition. The resultant MC algorithm ___RAR1MC___ is realized via block coordinate descent, whose analyses of convergence and computational complexity are produced. Experiments are conducted using synthetic random data, as well as real-world images with salt-and-pepper noise and multiple-input multiple-output radar signals in the presence of Gaussian mixture disturbances. In all three scenarios, the proposed algorithm outperforms the state-of-the-art robust MC methods in terms of recovery accuracy.

<img src="https://github.com/ShuDun23/t-Welsch-and-RAR1MC/blob/main/figures/Fig8.png" width="800px">

<img src="https://github.com/ShuDun23/t-Welsch-and-RAR1MC/blob/main/figures/table1.png" width="800px">

All experiments are conducted using MATLAB r2024b.

For real world images:

- To reproduce Figure 8, please run `main1.m`

- To reproduce Table III & IV, please run `main2.m`

For synthetic random data:

- we provide two detailed examples for reproducing Fig.5(b) and Fig.6(c). Please find `main_sigma1.m` and `main_sigma2.m`, respectively.

For detailed hyperparameter settings, please refer to the latest paper.

Should you have any questions, please feel free to reach out to Russell SHENG @ hnsheng2-c@my.cityu.edu.hk
