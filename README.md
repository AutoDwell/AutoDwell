# AutoDwell

**AutoDwell** is a deep neural network-based policy, aiming to dynamically schedule trains’ dwell time for the efficiency improvement of a metro system. To make a proper dwell decision to a train for its next station, the policy models spatio-temporal correlations of incoming passengers for all its related stations and the interactions between the train and trains on the same line. 

We propose to use deep reinforcement learning (DRL) framework to learn the policy. Taking advantage of DRL, we optimize the long-term rewards of dwell time settings which are formulated by weighting the passengers’ waiting time on platforms and the journey time on trains.

## Structure

Figure 1 (a) shows the architecture of the network, which consists of three components:

1. A *train feature extractor* to capture interactions between the current train and other trains on the same line based on the train state (see Figure 1(b) for the structure of the component).
2. A *passenger feature extractor* for embedding the upcoming passengers’ information among the passenger state by considering and weighing the ST correlations among all these subsequent stations of the train (the structure of the component can be found in Figure 2).
3. A *fusion network* to fuse the two parts of knowledge and accordingly provide Q-values for actions.

![](https://github.com/AutoDwell/AutoDwell/blob/master/img/1.png)

*Figure 1  (a) Overview of AutoDwell; (b) Structure of the train feature extractor*.

![](<https://github.com/AutoDwell/AutoDwell/blob/master/img/2.png>)

Figure 2: (a) An example of metro system. (b) Overview of the passenger feature extractor; (c) Structure of the transfer station learner.

## <!--Reference-->

<!--*Zhaoyuan Wang, Zheyi Pan. 2020. Shortening passengers’ travel time: A novel dynamic metro train.*-->

## <!--Author-->

<!--*Zhaoyuan Wang*-->



