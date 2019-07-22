# FGSM Attack against Neural Network

## Demostration

![an demo of FGSM](http://dl.guoyunzhe.net/adversarial-examples_fgsm_targeted_attack_7_3_0.1.png)

## Accuracy and Success Rate Evaluation (Targeted Attack)
 
In this section, we evaluate the accuracy score and the success rate of targeted fgsm attack.
We show these scores with different epsilon value (this table is a bit long).

| EPS Value| 0.000000| 0.025000| 0.050000| 0.075000| 0.100000| 0.125000| 0.150000| 0.175000| 0.200000| 0.225000| 0.250000| 0.275000| 0.300000| 0.325000| 0.350000| 0.375000| 0.400000| 0.425000| 0.450000| 0.475000| 0.500000 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Classification Accuracy| 0.958500| 0.787700| 0.331100| 0.110200| 0.036900| 0.013400| 0.007500| 0.005000| 0.005700| 0.004600| 0.005100| 0.004400| 0.005000| 0.003600| 0.003700| 0.003200| 0.003900| 0.004400| 0.004300| 0.003900| 0.004100 |
| Attacker's Success Rate| 0.000000| 0.097300| 0.363600| 0.549600| 0.632400| 0.676700| 0.694000| 0.693600| 0.705500| 0.693000| 0.690600| 0.684900| 0.688000| 0.687500| 0.669700| 0.682100| 0.667700| 0.672000| 0.673600| 0.662100| 0.672000 |

Plot these data:

![](http://dl.guoyunzhe.net/adversarial-examples_Figure_1.png)


