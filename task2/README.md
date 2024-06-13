Team BlackPearl PST's solution

Machine:
8 * A100 80G
186-core CPU
928G RAM

Running Time:

The estimated running time for train is 50 hours.
The estimated running time for inference is 60 hours.


**The theme of our entire solution is Grafted Learning**


In this competition, we abstracted the paper source tracing task into a pair-wise binary classification task. Therefore, a complete two-classification link is constructed based on multiple input information.

First, we can extract the contextual information about a reference in the paper just like the baseline (it is worth noting that there are certain errors in the label code and text extraction code in the baseline, which we will fix). According to this model, we can train a BERT two-classification model to achieve a basic effect.

The Trick which I used to train this bert:
  1. N-Gram MLM Pretrain in DBLP.json dataset
  2. EMA
  3. Rdrop
  4. Use a large-scale supervised data set annotated with rules to train a BERT as the input of the current BERT grafting learning
  5. TTA

Next, based on the bert obtained through training, Kfold cross-validation is performed on the training set to obtain the predicted probability of the training set and the predicted probability of the test set.

Then based on the title of the original paper and the title of the reference, a series of detailed information of the original text and reference is recalled from the DBLP.json paper library. At the same time, feature engineering was performed to list the total citations of each author, the total citations of journals, and the total citations of the author's institution to represent the value of the author, journal, and institution.

Finally, all the information is spliced ​​into a long text, LLM is used to make binary classification judgments, and the final probability is output.

The final training and inference text is as follows:

```
You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper.The following points define whether a reference is a source paper:
Is the main idea of paper p inspired by the reference？
Is the core method of paper p derived from the reference？
Is the reference essential for paper p? Without the work of this reference, paper p cannot be completed.
The number of this reference paper is 0, and the name of this reference paper is <stochastic blockmodel approximation of a graphon: theory and consistent estimation>.The rank of this paper is 3, total rank is 52
The number of this reference paper is 0, and the name of this reference paper is <stochastic blockmodel approximation of a graphon: theory and consistent estimation>.The rank of this paper is 3, total rank is 52
The number of this reference paper is 0, and the name of this reference paper is <stochastic blockmodel approximation of a graphon: theory and consistent estimation>.The rank of this paper is 3, total rank is 52
Title:
G-Mixup: Graph Data Augmentation for Graph Classification
Abstract:
This work develops mixup for graph data. Mixup has shown superiority in improving the generalization and robustness of neural networks by interpolating features and labels between two random samples. Traditionally, Mixup can work on regular, grid-like, and Euclidean data such as image or tabular data. However, it is challenging to directly adopt Mixup to augment graph data because different graphs typically: 1) have different numbers of nodes; 2) are not readily aligned; and 3) have unique typologies in non-Euclidean space. To this end, we propose G-Mixup to augment graphs for graph classification by interpolating the generator (i.e., graphon) of different classes of graphs. Specifically, we first use graphs within the same class to estimate a graphon. Then, instead of directly manipulating graphs, we interpolate graphons of different classes in the Euclidean space to get mixed graphons, where the synthetic graphs are generated through sampling based on the mixed graphons. Extensive experiments show that G-Mixup substantially improves the generalization and robustness of GNNs.
Body:
正文
References:
参考文献123456
The number of this reference paper is 0, and the name of this reference paper is <stochastic blockmodel approximation of a graphon: theory and consistent estimation>.The rank of this paper is 3, total rank is 52
The number of this reference paper is 0, and the name of this reference paper is <stochastic blockmodel approximation of a graphon: theory and consistent estimation>.The rank of this paper is 3, total rank is 52
The number of this reference paper is 0, and the name of this reference paper is <stochastic blockmodel approximation of a graphon: theory and consistent estimation>.The rank of this paper is 3, total rank is 52

In addition, the year of source paper is 2022, the keywords of source paper is , the venue of source paper is International Conference on Machine Learning, the cite of source paper is 105, the cite of source paper's venue is 80621,this reference paper's first author's name is Edoardo M Airoldi, total cite is 0, organization is Harvard University, total cite of this organization is 175222,this reference paper's second author's name is Thiago B Costa, total cite is 209, organization is Harvard University, total cite of this organization is 175222,this reference paper's third author's name is Stanley H Chan, total cite is 3291, organization is Harvard University, total cite of this organization is 175222,the year of this reference paper is 2013, the venue of this reference paper is neural information processing systems, the keywords of this reference paper is stochastic blockmodel approximation graphon consistent estimation, the cite of this refenence paper is 209, the cite of reference paper's venue is 100116, the appearance time of this reference paper is 6, the abstract of this reference paper is:Non-parametric approaches for analyzing network data based on exchangeable graph models (ExGM) have recently gained interest. The key object that defines an ExGM is often referred to as a graphon. This non-parametric perspective on network modeling poses challenging questions on how to make inference on the graphon underlying observed network data. In this paper, we propose a computationally efficient procedure to estimate a graphon from a set of observed networks generated from it. This procedure is based on a stochastic blockmodel approximation (SBA) of the graphon. We show that, by approximating the graphon with a stochastic block model, the graphon can be consistently estimated, that is, the estimation error vanishes as the size of the graph approaches infinity..
```

In the end we used three different LLMs for fusion, The following is the specific A\B list score of each step of our model：

Bert:
leaderboard score: 0.470,   privateboard score: 0.417

ChatGLM1(addcite):
leaderboard score: 0.525,   privateboard score: 0.477

ChatGLM2(addcite_addbib):
leaderboard score: 0.530,   privateboard score: 0.481

ChatGLM3(addcite_mistral_rag):
leaderboard score: 0.520,   privateboard score: 0.472

Fianl result:
private board score: 0.488

inference & train scripts
```
torchrun --nproc_per_node 8 step0_pretrain_0.py
torchrun --nproc_per_node 8 step0_pretrain_1.py
python step1_build_bert_input.py
python step2_train_bert.py
python step3_build_llm_input.py
sh mistral_embed_inference.sh
python step4_rag.py
sh train.sh
sh inference_0_0.sh
sh inference_0_1.sh
sh inference_1_0.sh
sh inference_1_1.sh
sh inference_2_0.sh
sh inference_2_1.sh
python merge.py
```
