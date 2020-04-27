## Deep Learning for Anomaly Detection

> This repo contains code for experiments we run at Cloudera Fast Forward for implementing deep learning for anomaly detection usecases. We include implementations of 4 neural networks () in Tensorflow 2.0. We have released a report detailing the technical details for each approach in our online report [here](https://ff12.fastforwardlabs.com/). An interactive visualization of some results can be found [here](http://blip.fastforwardlabs.com/)

Anomalies, often referred to as outliers, abnormalities, rare events, or deviants, are data points or patterns in data that do not conform to a notion of normal behavior. Anomaly detection, then, is the task of finding those patterns in data that do not adhere to expected norms, given previous observations. The capability to recognize or detect anomalous behavior can provide highly useful insights across industries. Flagging unusual cases or enacting a planned response when they occur can save businesses time, costs, and customers. Hence, anomaly detection has found diverse applications in a variety of domains, including IT analytics, network intrusion analytics, medical diagnostics, financial fraud protection, manufacturing quality control, marketing and social media analytics, and more.


## Structure of Repo

-- train.py
-- data
    - kdd
-- models
    -- ae.py
    -- vae.py
    -- seq2seq.py
    -- gan.py


To train the currently implemented model, run

`python3 train.py`

## Generating Benchmark Results
### KDD99 Dataset
