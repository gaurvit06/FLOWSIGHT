# Zero-day experiment (leave one attack out)

Classifier: xgb. Unknown-attack rule: the classifier says Benign but either the anomaly detector (Isolation Forest + range rule, benign-only) fires or the classifier is less than 90% sure. It raises false alarms on **0.64%** of benign test flows (mean over the runs).

| held-out attack   |   test flows | caught by classifier (as another attack)   | caught only by unknown-attack rule   | total caught   | missed   |
|:------------------|-------------:|:-------------------------------------------|:-------------------------------------|:---------------|:---------|
| BruteForce        |         1372 | 13.3%                                      | 9.0%                                 | 22.3%          | 77.7%    |
| DDoS              |        19202 | 60.3%                                      | 2.7%                                 | 63.0%          | 37.0%    |
| DoS               |        29062 | 1.5%                                       | 3.8%                                 | 5.3%           | 94.7%    |
| PortScan          |        13604 | 0.6%                                       | 0.3%                                 | 0.9%           | 99.1%    |

*Caught by classifier* means the model had never seen this attack but still called it an attack (a different type). *Caught only by unknown-attack rule* is what the rule adds on top.