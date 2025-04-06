# EVO: Project Proposal
#### **Group Members:** Yahya Rahhawi, James Cummings, Jiffy Lesica, Lukka Wolff

### Abstract
Using principles of evolution to build a machine learning algorithm and explore its capabilities compared to gradient descent, we will focus primarily on its ability to solve non-differentiable problems. Explore different changes to the evolutionary principles and compare the performance with those changes: adjust mutation rates, mutation magnitude, fitness criteria, and population management. Through various explorations such as symbolic regression, handwriting recognition, and evolving feature selection, we will demonstrate the strengths and weaknesses of applying evolutionary principles to machine learning. 

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

One risk is that the evolutionary algorithm we design may not perform well enough to produce meaningful results, especially when compared with traditional methods like gradient descent. We can mitigate this risk by clarifying that this research is an exploration, rather than a test of a specific hypothesis: we are not trying to prove that evolutionary algorithms perform better than gradient descent in non-differentiable problems, but instead exploring how they behave in these scenarios in comparison to gradient descent. For example, if our algorithm fails to output accurate predictions MNIST digit classification task - or "more accurate" results than gradient descent - we may struggle to show the benefits of using evolution-based approaches. This could happen if our method is too slow to improve, or fails to make progress over time.

Another key risk is the amount of time and computing power required to run evolutionary experiments. Evolutionary algorithms often need to evaluate many different possible solutions over hundreds or thousands of generations (such as seen here: https://www.youtube.com/watch?v=N3tRFayqVtk), which can be very time-consuming. If we’re limited by the speed of our computers or the availability of computing resources, we may not be able to run enough experiments to tune the algorithm or explore all the variations we planned (like testing different mutation strategies or selection criteria). This could limit the depth and scope of our final results.

### Ethics Statement

### Tentative Timeline
- Week 1: (Current) Project proposal.
- Week 2: Baseline algorithm implementation.
- Week 3: Begin Symbolic Regression and MNIST problems.
- Week 4: Finish Begin Symbolic Regression and MNIST problems.
- Week 5: Evolving Feature selection and alternative implementation exploration. Finalize project.
