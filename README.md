# Sensorimotor Learning with Stability Guarantees via Autonomous Neural Dynamic Policies

Paper submitted to the Special Issue on "Learning and Intelligent Optimization" of the ACM Transactions on Evolutionary Learning and Optimization (TELO) journal.

![Screenshot from 2024-01-10 15-45-20](https://github.com/NOSALRO/andps/assets/50770773/78e57186-d31e-46d6-8027-3f3df28a995e)

Abstract
---------------

State-of-the-art sensorimotor learning algorithms, either in the context of reinforcement learning or imitation learning, offer policies that can often produce unstable behaviors, damaging the robot and/or the environment. Moreover, it is very difficult to interpret the optimized controller and analyze its behavior and/or performance. Traditional robot learning, on the contrary, relies on dynamical system-based policies that can be analyzed for stability/safety. Such policies, however, are neither flexible nor generic and usually work only  with proprioceptive sensor states. In this work, we bridge the gap between generic neural network policies and dynamical system-based policies, and we introduce Autonomous Neural Dynamic Policies (ANDPs) that: (a) are based on autonomous dynamical systems, (b) always produce asymptotically stable behaviors, and (c) are more flexible than traditional stable dynamical system-based policies. ANDPs are fully differentiable, flexible generic-policies that can be used for both imitation learning and reinforcement learning setups, while ensuring asymptotic stability. Through several experiments, we explore the flexibility and capacity of ANDPs in several imitation learning tasks including experiments with image observations. The results show that ANDPs combine the benefits of both neural network-based and dynamical system-based methods.
