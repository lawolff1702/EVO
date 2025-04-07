# EVO: Project Proposal
#### **Group Members:** Yahya Rahhawi, James Cummings, Jiffy Lesica, Lukka Wolff

### Abstract
Using principles of evolution to build a machine learning algorithm and explore its capabilities compared to gradient descent - we will focus primarily on its ability to solve non-differentiable problems. Explore different changes to the evolutionary principles and compare the performance with those changes: adjust mutation rates, mutation magnitude, fitness criteria, and population management. Through various explorations such as symbolic regression, handwriting recognition, and evolving feature selection, we will demonstrate the strengths and weaknesses of applying evolutionary principles to machine learning. 

### Motivation and Question

We want to address implementation of niche machine learning algorithms and their applications. In our class lectures we often talk about finding the local minima of functions and how having convex functions aids us in finding this point. In models that rely on gradient descent we rely on being able to differentiate the loss function. **What happens when we don't have a convex or differentiable loss function?** *Evolutionary and genetic algorithms* present an alternative optimization framework that does not require gradient information.

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

To complete our project, we will need access to several datasets, personal and cloud computing, and Python libraries for implementation and experimentation. For the application side of our data, we will use three publicly available datasets. We will use [Feynman Dataset](https://space.mit.edu/home/tegmark/aifeynman.html) for symbolic regression, which contains input-output data derived from real physics equations. For our handwriting recognition we will use the [MNIST Handwritten Digit Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv). For feature selection and binary classification we will explore the [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html). All of these datasets are freely available! Our models may be computationally intensive, however, we expect that our laptops should be able to handle it. If computation proves too complex cloud-based Jupyter notebooks like Google Colab or Middlebury's ADA Cluster can also be used.

We will be implementing our own evolutionary algorithm from scratch in Python using NumPy, pandas, and PyTorch. We will also use matplotlib or SeaBorn for data visualization and scikit-learn for model evaluation and comparison. These are all free!

### What You Will Learn

In this project, group members will deepen their understanding of evolutionary machine learning algorithms, focusing on how evolutionary principles compare to or complement gradient descent methods in non-differentiable problem contexts. As such, part of this project will involve familiarization with the complexities and real-life examples of non-differentiable problems. We will gain hands-on experience implementing evolution-inspired ML techniques, including adjusting mutation rates and weights, and defining selection criteria tailored for given sub-problems (such as symbolic regression or MNIST handwriting recognition).

Group members will also enhance their proficiency in Python programming using machine learning libraries such as PyTorch. The team will strengthen project management skills involving task designation and completion tracking on Github. Further, we will refine our understanding of collaborative workflows by developing our project on local git branches. Finally, participants will critically assess algorithmic performance - i.e. results of data analysis for non-differentiable problems using gradient descent vs. evolutionary algorithms - and effectively communicate the strengths, limitations, and appropriate applications of evolutionary machine learning techniques.

### Risk Statement

One risk is that the evolutionary algorithm we design may not perform well enough to produce meaningful results, especially when compared with traditional methods like gradient descent. We can mitigate this risk by clarifying that this research is an exploration, rather than a test of a specific hypothesis: we are not trying to prove that evolutionary algorithms perform better than gradient descent in non-differentiable problems, but instead exploring how they behave in these scenarios in comparison to gradient descent. For example, if our algorithm fails to output accurate predictions MNIST digit classification task - or "more accurate" results than gradient descent - we may struggle to show the benefits of using evolution-based approaches. This could happen if our method is too slow to improve, or fails to make progress over time.

Another key risk is the amount of time and computing power required to run evolutionary experiments. Evolutionary algorithms often need to evaluate many different possible solutions over hundreds or thousands of generations (such as seen here: https://www.youtube.com/watch?v=N3tRFayqVtk), which can be very time-consuming. If we’re limited by the speed of our computers or the availability of computing resources, we may not be able to run enough experiments to tune the algorithm or explore all the variations we planned (like testing different mutation strategies or selection criteria). This could limit the depth and scope of our final results.

### Ethics Statement

As part of our machine learning project exploring evolutionary optimization techniques,
we recognize the importance of addressing ethical considerations that relate to the
potential impacts of our work. Below are reflections based on the core questions on the assignment:


##### Who could benefit from our project?

Our project brings together evolutionary algorithms and core machine learning tasks like classification, symbolic regression, and feature selection. By doing this, we hope to make machine learning not only more effective but also more interpretable and accessible.

This work could benefit several groups. Machine learning researchers might find value in the way we explore optimization beyond traditional gradient-based methods. Educators and students could use our project as a hands-on, engaging example of how bio-inspired algorithms can be applied in real-world ML problems. And if we apply our techniques to datasets like the breast cancer classification dataset, there’s potential to support the medical and health research community by improving how we select features — possibly leading to more accurate and streamlined diagnostic tools.

##### Who could be excluded or harmed?

While our project has a lot of potential, we also recognize that there are some important risks to keep in mind.

For example, if we use datasets like the breast cancer dataset, there’s a chance the data may not fully represent all populations. This could lead to biased models that don’t perform as well for certain demographic groups, unintentionally excluding or even harming people who are already underrepresented.

Another concern is that evolved models — especially ones generated through symbolic regression — might appear deceptively simple or elegant. This can lead people to place too much trust in their accuracy or fairness, even when that trust isn’t justified.

Finally, if we don’t make the project accessible through clear explanations or visualizations, people without a strong background in machine learning or computer science might find it difficult to engage with or benefit from our work.

##### Will the world be a better place because of our project?

Like any research project, We believe it will,  based on the following assumptions:

Evolutionary algorithms can enhance machine learning in areas where traditional methods often fall short. They’re especially useful for optimizing problems that aren’t easily handled by gradient-based approaches, and they can help discover symbolic models that are not only effective but also interpretable.

By making machine learning more interpretable and inspired by natural processes, we can help demystify the field and make it more accessible to a broader range of people — especially students and researchers coming from interdisciplinary or non-traditional backgrounds. In many real-world ML applications, it’s not just about how accurate a model is — it’s about understanding how and why it makes decisions. This level of transparency is crucial when we want to apply models in settings that demand accountability.


##### Addressing potential algorithmic bias

If we end up using medical datasets, such as one for breast cancer classification, we’ll take extra care to examine how fair and representative our models are.

We plan to look closely at the demographic distribution of the dataset and note any imbalances, whether they relate to race, gender, age, or other factors. These kinds of gaps can lead to biased models that don’t serve all groups equally. In addition to tracking overall accuracy, we’ll also report more meaningful performance metrics like false positive and false negative rates. This is especially important in medical applications, where a single misclassification could lead to real-world harm.


### Tentative Timeline
- Week 1: (Current) Project proposal.
- Week 2: Baseline algorithm implementation.
- Week 3: Begin Symbolic Regression and MNIST problems.
- Week 4: Finish Begin Symbolic Regression and MNIST problems.
- Week 5: Evolving Feature selection and alternative implementation exploration. Finalize project.
