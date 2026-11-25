# Class-imbalance ablation (XGBoost, validation split)

|               |   macro-F1 |   recall Benign |   recall BruteForce |   recall DDoS |   recall DoS |   recall PortScan |
|:--------------|-----------:|----------------:|--------------------:|--------------:|-------------:|------------------:|
| none          |     0.996  |          0.9993 |              0.9931 |        0.998  |       0.9984 |            0.9945 |
| class weights |     0.9955 |          0.9992 |              0.9931 |        0.998  |       0.9984 |            0.9972 |
| SMOTE         |     0.9962 |          0.9993 |              0.9931 |        0.9975 |       0.999  |            0.9965 |

Look at the smallest classes' recall, not macro-F1 alone.
