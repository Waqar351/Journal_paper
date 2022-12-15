# SMM: A Simple, Efficient and Accurate Quantification Algorithm
*Waqar Hassan, André Maletzke,  Gustavo Batista*

## The experiments were conducted with 19 benchmark datasets using 23 quantifiers.
We learn quantifiers on different training samples and each sample contains different class distributions. We vary the training class distribution as `[0.05, 0.1, 0.3, 0.5, 0.7, 0.9]`. Similarly, we also vary the test set size ranges from 10 to 500 and for each test set size, we vary the class distribution as `[0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]`.

In addition, we also access the inference time of quantification methods. We measure the runtime of each quantifier on different sample size ranges from 100 to 100000 and take the average runtime across different sizes.

To access the quantifiers performance, we use `MAE` and `MRE` as a measure. For statistical test we use post-hoc Neymani test.

Below are the results generated from our experiments.

## Runtime vs Accuracy performance

<div>
<p align= "center">
      <img src = "Figures/Fig_instXrank.pdf" title = "A time and accuracy plot">
</p>
</div>

## Overall accuracy of quantifiers

## Influence of training class proportion

## Influence of distribution shift

## Influence of test set size
We shows the results when test set size is small (composed of 10 instances) and when the test set size is large (composed of 500 instances).
The results are displayed using box plots in terms of MAE and MRE along with Critical difference diagram evaluated using both measures.

### Small test set size


### Large test set size

# Journal_paper
This project deals with a supervised learning task: Quantification whose objective is to estimate the class distribution in a test sample.

NOTE : To run experiment for any method use the following in the command prompt

            `python Experiment_new_setup.py DatasetName MethodName --n_iterations`
            
      E.g., **python Experiment_new_setup.py spambase smm --it=10**
