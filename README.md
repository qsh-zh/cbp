# CBP(Constrained Belief Propagation)

Constrained Belief Progapation is a package for solving inference tasks with collective\aggregate evidence.

## What is collective\aggregate evidence?

Let us consider a task, estimating dymanics of bird migration. We consider the following different kinds of evidence



1. Install trackers for interested birds. So we access to the trajectories of sampled birds. We can query the position of interested birds at any time and how they move.



2. We shoot a video for the birds migration. We can quety the distibution of the birds in the space at any time. However, we do not have access to individual trajectories and how birds move overtime.



The second information is an instance of collective\aggregate evidence, which remove the distinguishability of individuals. 



## Why do we need collective\aggregate evidence?



* Easy and cheap to acquire

* Privacy concern



Sometimes, collective\aggregate evidence is the ony information we access to analysis collective behavior.



## How can we represent the special evidence?



In CBP, we represent the special evidence as a special nodes  in probabilistic graphical model(PGM), named node with fixed marginal constraints.  A hidden markov model with  aggregate evidence can be presented as: 

![image-20200716201949266](https://i.imgur.com/iuuxhT7.png)

The shaded nodes represents the aggregate evidence.  We can introduce a simple version:

![image-20200716202622245](https://i.imgur.com/W7yfLFN.png)

## How to do inference?

In CBP framework, we present a nice elegant connections between multi-marginal optimal transport problems and inference problem for PGM.  We implement algorithms similar to the standard belief propagation. In CBP, we present `Iterative Scaling Belief Propagation` and `Contrained Norm-product` two algorithms. More details can be find the [paper](https://arxiv.org/pdf/2006.14113.pdf).

## Citation

```
@article{haasler2020multi,
  title={Multi-marginal optimal transport and probabilistic graphical models},
  author={Haasler, Isabel and Singh, Rahul and Zhang, Qinsheng and Karlsson, Johan and Chen, Yongxin},
  journal={arXiv preprint arXiv:2006.14113},
  year={2020}
}
```

