# Sensorimotor Learning with Stability Guarantees via Autonomous Neural Dynamic Policies

**Authors:** *Dionis Totsila\*, Konstantinos Chatzilygeroudis\*, Valerio Modugno, Denis Hadjivelichkov, and Dimitrios Kanoulas*

\* Equal contribution

![Screenshot from 2024-01-10 15-45-20](https://github.com/NOSALRO/andps/assets/50770773/78e57186-d31e-46d6-8027-3f3df28a995e)

## Abstract

State-of-the-art sensorimotor learning algorithms, either in the context of reinforcement learning or imitation learning, offer policies that can often produce unstable behaviors, damaging the robot and/or the environment. Moreover, it is very difficult to interpret the optimized controller and analyze its behavior and/or performance. Traditional robot learning, on the contrary, relies on dynamical system-based policies that can be analyzed for stability/safety. Such policies, however, are neither flexible nor generic and usually work only with proprioceptive sensor states. In this work, we bridge the gap between generic neural network policies and dynamical system-based policies, and we introduce Autonomous Neural Dynamic Policies (ANDPs) that: (a) are based on autonomous dynamical systems, (b) always produce asymptotically stable behaviors, and (c) are more flexible than traditional stable dynamical system-based policies. ANDPs are fully differentiable, flexible generic-policies that can be used for both imitation learning and reinforcement learning setups, while ensuring asymptotic stability. Through several experiments, we explore the flexibility and capacity of ANDPs in several imitation learning tasks including experiments with image observations. The results show that ANDPs combine the benefits of both neural network-based and dynamical system-based methods.

## Citing ANDPs

If you use this code in a scientific publication, please use the following citation:

```bibtex
@article{dionis2024andps,
        title={{Sensorimotor Learning with Stability Guarantees via Autonomous Neural Dynamic Policies}},
        author={Totsila, Dionis and Chatzilygeroudis, Konstantinos and Modugno, Valerio and Hadjivelichkov, Denis and Kanoulas, Dimitrios},
        year={2024},
        journal={{Preprint}}
      }
```

## Acknowledgments

Konstantinos Chatzilygeroudis was supported by the [Hellenic Foundation for Research and Innovation](https://www.elidek.gr/en/homepage/) (H.F.R.I.) under the "3rd Call for H.F.R.I. Research Projects to support Post-Doctoral Researchers" (Project Acronym: [NOSALRO](https://nosalro.github.io/), Project Number: 7541). Dimitrios Kanoulas and Valerio Modugno were supported by the UKRI Future Leaders Fellowship [MR/V025333/1] (RoboHike).

<p align="center">
<img src="https://www.elidek.gr/wp-content/themes/elidek/images/elidek_logo_en.png" alt="logo_elidek"/>
<img src="https://www.cinuk.org/content/uploads/2022/11/UKRI-logo2.png" alt="logo_ukri" width="25%"/>
<p/>

This work was conducted as collaboration of the [Computational Intelligence Lab](http://cilab.math.upatras.gr/) (CILab), Department of Mathematics, University of Patras, Greece, and the [Robot Perception and Learning Lab](https://rpl-as-ucl.github.io/) (RPL Lab), Department of Computer
Science, University College London (UCL), United Kingdom.

<p align="center">
<img src="https://nosalro.github.io/images/logo_cilab.jpg" alt="logo_cilab" width="50%"/>
<img src="https://www.upatras.gr/wp-content/uploads/up_2017_logo_en.png" alt="logo_cilab" width="50%"/>
<img src="https://rpl-as-ucl.github.io/images/logos/rpl-cs-ucl-logo.png" alt="logo_rpl" width="50%"/>
</p>

## License

[BSD 2-Clause &#34;Simplified&#34; License](https://opensource.org/license/bsd-2-clause/)
