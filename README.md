## Deep Learning for Anomaly Detection

<img src="images/testresult.png" width="100%">

> This repo contains code for experiments we have run at Cloudera Fast Forward for implementing deep learning for anomaly detection usecases. 

We include implementations of several neural networks (Autoencoder, Variational Autoencoder, Bidirectional GAN, Sequence Models) in Tensorflow 2.0 and two other baselines (One Class SVM, PCA). We have released a report detailing the technical details for each approach in our online report [here](https://ff12.fastforwardlabs.com/). An interactive visualization of some results can be found [here](http://blip.fastforwardlabs.com/)

Anomalies, often referred to as outliers, abnormalities, rare events, or deviants, are data points or patterns in data that do not conform to a notion of normal behavior. Anomaly detection, then, is the task of finding those patterns in data that do not adhere to expected norms, given previous observations. The capability to recognize or detect anomalous behavior can provide highly useful insights across industries. Flagging unusual cases or enacting a planned response when they occur can save businesses time, costs, and customers. Hence, anomaly detection has found diverse applications in a variety of domains, including IT analytics, network intrusion analytics, medical diagnostics, financial fraud protection, manufacturing quality control, marketing and social media analytics, and more.


## Structure of Repo

```bash
├── data
│   ├── kdd
│   │   ├── all.csv
│   │   ├── train.csv
│   │   ├── test.csv
├── models
│   ├── ae
│   ├── bigan
│   ├── ocsvm
│   ├── vae
├── train.py 
```
 
To train the currently implemented models, run

`python3 train.py`

This checks for the available datasets (downloads them as needed), trains and evaluates implemented models.

## Generating Benchmark Results

TODO: MLFlow instrumentation to track global metrics for a single view of all training operations.

### Running on Cloudera Machine Learning

TODO: Map application abstraction for easy deployment on Cloudera Machine Learning.

## TODO
- Implement evaluation test harness
    - flow for evaluating trained model and generating performance charts.
    - integrate with MLFlow to easily visualize details for all runs
- Map to CML
    - Export models from each run for use in CML application
    - CML model/application to host and interact with model endpoint  
- Update Repo with insights from all runts
- Stretch: Include image dataset.
 
