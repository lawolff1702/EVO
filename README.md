# EVO: Project Proposal
#### **Group Members:** Yahya Rahhawi, James Cummings, Jiffy Lesica, Lukka Wolff

### Abstract
Using principles of evolution to build a machine learning algorithm and explore its capabilities compared to gradient descent - we will focus primarily on its ability to solve non-differentiable problems. Explore different changes to the evolutionary principles and compare the performance with those changes: adjust mutation rates, mutation magnitude, fitness criteria, and population management. Through various explorations such as symbolic regression, handwriting recognition, and evolving feature selection, we will demonstrate the strengths and weaknesses of applying evolutionary principles to machine learning. 

### Motivation and Question

We want to address implementation of niche machine learning algorithms and their applications. In our class lectures we often talk about finding the local minima of functions and how having convex functions aids us in finding this point. In models that rely on gradient descent we rely on being able to differentiate the loss function. **What happens when we don't have a convex or non-differentiable loss function?** *Evolutionary and genetic algorithms* present an alternative optimization framework that does not require gradient information.

Our project will be three-fold. First, we will focus on handwriting recognition using the MNIST dataset. While this is a more traditional classification problem, it provides us the opportunity to test how evolutionary algorithms perform against standard gradient descent in a high-dimensional, image-based context.

Our next focus is on building a symbolic regression system using evolutionary principles to evolve mathematical expressions that fit data. We will test this approach on the Feynman dataset, which consists of real physics equations and their corresponding input-output data. Since the equations are unknown to the model, the task is to rediscover them purely through data-driven search. 

Finally, we will explore evolutionary feature selection using the Breast Cancer Wisconsin dataset. In this task, the evolutionary algorithm will be used to identify the most relevant subset of features for classification. Because feature selection is a discrete and combinatorial problem, it will hopefully serve as a strong example of where evolutionary methods can shine.


### Planned Deliverables

- Implementation of Evolution based machine learning algorithm
- Application of algorithm to Symbolic Regression problem
- Application of algorithm to MNIST handwriting recognition problem
- Evolving feature selection on classification (Breast Cancer dataset)
- Comparison of different implementations of evolution: evolution based feature selection, diversity preservation principles, island models, etc

### Resources Required

### What You Will Learn

In this project, group members will deepen their understanding of evolutionary machine learning algorithms, focusing on how evolutionary principles compare to or complement gradient descent methods in non-differentiable problem contexts. As such, part of this project will involve familiarization with the complexities and real-life examples of non-differentiable problems. We will gain hands-on experience implementing evolution-inspired ML techniques, including adjusting mutation rates and weights, and defining selection criteria tailored for given sub-problems (such as symbolic regression or MNIST handwriting recognition).

Group members will also enhance their proficiency in Python programming using machine learning libraries such as PyTorch. The team will strengthen project management skills involving task designation and completion tracking on Github. Further, we will refine our understanding of collaborative workflows by developing our project on local git branches. Finally, participants will critically assess algorithmic performance - i.e. results of data analysis for non-differentiable problems using gradient descent vs. evolutionary algorithms - and effectively communicate the strengths, limitations, and appropriate applications of evolutionary machine learning techniques.

### Risk Statement

### Ethics Statement

### Tentative Timeline
- Week 1: (Current) Project proposal.
- Week 2: Baseline algorithm implementation.
- Week 3: Begin Symbolic Regression and MNIST problems.
- Week 4: Finish Begin Symbolic Regression and MNIST problems.
- Week 5: Evolving Feature selection and alternative implementation exploration. Finalize project.
