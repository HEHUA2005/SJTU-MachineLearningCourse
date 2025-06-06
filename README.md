# How to run.
First unzip the data `xxxx.xlsx` to Dataset/.

It can be downloaded in [Dataset.zip](https://oc.sjtu.edu.cn/courses/77474/files/11006511?wrap=1)

## For pre-processing.
```bash
PYTHONPATH=Dataset/ python pre_process.py
```
## For reproducing the experiment.
```bash
PYTHONPATH=. python task*/**.py
```