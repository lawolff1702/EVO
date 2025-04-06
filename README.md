# EVO: Project Proposal
#### **Group Members:** Yahya Rahhawi, James Cummings, Jiffy Lesica, Lukka Wolff

### Abstract
Using principles of evolution to build a machine learning algorithm and explore its capabilities compared to gradient descent - we will focus primarily on its ability to solve non-differentiable problems. Explore different changes to the evolutionary principles and compare the performance with those changes: adjust mutation rates, mutation magnitude, fitness criteria, and population management. Through various explorations such as symbolic regression, handwriting recognition, and evolving feature selection, we will demonstrate the strengths and weaknesses of applying evolutionary principles to machine learning. 

### Motivation and Question

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


As part of our machine learning project exploring evolutionary optimization techniques,
we recognize the importance of addressing ethical considerations that relate to the
potential impacts of our work. Below are reflections based on the core questions on the assignment:

---

### Who could benefit from our project?

Our project combines bio-inspired evolutionary algorithms with core ML tasks such as
classification, symbolic regression, and feature selection. The following groups stand
to benefit:

- **Machine learning researchers** seeking novel, interpretable
  optimization techniques beyond gradient descent.
- **Students and educators**, as our implementation can serve as an engaging and
  educational example of evolutionary computation applied to ML.
- **Medical and health research communities**, particularly if we apply evolutionary
  feature selection to datasets like the breast cancer dataset, potentially leading to
  more accurate and concise diagnostic models.

---

### Who could be excluded or harmed?

We are aware of risks and limitations associated with our approach:

- **Bias in datasets**: Using datasets such as breast cancer data may result in biased
  models if the data lacks demographic diversity, leading to exclusion or harm to
  underrepresented groups.
- **Overtrust in evolved models**: Symbolic regression models or evolved classifiers may
  be interpreted as more accurate or fair than they truly are, particularly if their
  simplicity is mistaken for correctness.
- **Accessibility concerns**: Users without a CS or ML background may be excluded from
  benefiting if we do not provide explanations, visualizations, or clear documentation.

### Will the world be a better place because of our project?

Like any research project, We believe it will,  based on the following two assumptions:

1. **Evolutionary algorithms can improve ML outcomes** where traditional methods struggle,
    for example, optimizing non-differentiable objectives or discovering symbolic models with
   interpretable structure.

2. **Making ML more interpretable and nature-inspired** through evolutionary design can
   broaden understanding and interest in ML, particularly for learners and researchers
   from non-traditional or interdisciplinary backgrounds. In ML, we care more about how the model makes decisions rather than the correctness of these decisions to make sure
   we can reliably deploy it in scenarios where accountability is required.


### Addressing potential algorithmic bias

If we use datasets for medical classification:

- We will examine the **demographic distribution** of the dataset and document any
  potential imbalance (e.g., race, gender, age).
- We will report metrics beyond accuracy, including **false positives and false negatives**,
  especially in medical contexts where misclassification can have serious consequences.


### Tentative Timeline
- Week 1: (Current) Project proposal.
- Week 2: Baseline algorithm implementation.
- Week 3: Begin Symbolic Regression and MNIST problems.
- Week 4: Finish Begin Symbolic Regression and MNIST problems.
- Week 5: Evolving Feature selection and alternative implementation exploration. Finalize project.
