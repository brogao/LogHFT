In particular, the package depends on the following requirements. Note that regex matching in Python is brittle, so we recommend fixing the regex library to version 2022.3.2.

+ python 3.8+
+ regex 2023.10.3
+ numpy
+ pandas
+ scipy
+ scikit-learn


### Get started

1. Run the benchmark:
  
    For each log parser, we provide a benchmark script to run log parsing on the [loghub_24 datasets](https://github.com/logpai/logparser/tree/main/data#loghub_2k) for evaluating parsing accuarcy. You can also use [other benchmark datasets for log parsing](https://github.com/logpai/logparser/tree/main/data#datasets).

    ```
    cd logparser/LogHFT
    python benchmark.py
    ```
