import sys
sys.path.append("../../")
from logparser.LogHFT import LogParser
from logparser.utils import evaluator
import os
import pandas as pd
import time

input_dir = "../../data/loghub_24/"  # The input directory of log file
output_dir = "Log_result/"  # The output directory of parsing results

benchmark_settings = {
    "HDFS": {
        "log_file": "HDFS/HDFS_2k.log",
        "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "regex": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"],
        "st": 0.7,
        "splitters": r'(\s+)',
        "ngram_threshold": 0.9,
        "n": 3,
        "parsing_method": "seqDist",
    },
    "Hadoop": {
        "log_file": "Hadoop/Hadoop_2k.log",
        "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
        "regex": [r"(\d+\.){3}\d+"],
        "st": 0.65,
        "splitters": r'(\s+|\[|\])',
        "ngram_threshold": 0.6,
        "n": 4,
        "parsing_method": "seqDist",
    },
    "Spark": {
        "log_file": "Spark/Spark_2k.log",
        "log_format": "<Date> <Time> <Level> <Component>: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"\b[KGTM]?B\b", r"([\w-]+\.){2,}[\w-]+"],
        "st": 0.6,
        "splitters": r'(\s+|\[|\]|\(|\))',
        "ngram_threshold": 0.8,
        "n": 3,
        "parsing_method": "seqDist",
    },
    "Zookeeper": {
        "log_file": "Zookeeper/Zookeeper_2k.log",
        "log_format": "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
        "regex": [r"(/|)(\d+\.){3}\d+(:\d+)?"],
        "st": 0.9,
        "splitters": r'(\s+)',
        "ngram_threshold": 0.8,
        "n": 3,
        "parsing_method": "seqDist",
    },
    "BGL": {
        "log_file": "BGL/BGL_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        "regex": [r"core\.\d+"],
        "st": 0.7,
        "splitters": r'(\s+|\(|\)|=)',
        "ngram_threshold": 0.4,
        "n": 3,
        "parsing_method": "ngram",
    },
    "HPC": {
        "log_file": "HPC/HPC_2k.log",
        "log_format": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
        "regex": [r"=\d+"],
        "st": 0.8,
        "splitters": r'(\s+|\(|\))',
        "ngram_threshold": 0.8,
        "n": 3,
        "parsing_method": "seqDist",
    },
    "Thunderbird": {
        "log_file": "Thunderbird/Thunderbird_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>",
        "regex": [r"(\d+\.){3}\d+"],
        "st": 0.6,
        "splitters": r'(\s+)',
        "ngram_threshold": 0.5,
        "n": 2,
        "parsing_method": "seqDist",
    },
    "Windows": {
        "log_file": "Windows/Windows_2k.log",
        "log_format": "<Date> <Time>, <Level>                  <Component>    <Content>",
        "regex": [r"0x.*?\s"],
        "st": 0.85,
        "splitters": r'(\s+|\{|\})',
        "ngram_threshold": 0.6,
        "n": 3,
        "parsing_method": "combined",
    },
    "Linux": {
        "log_file": "Linux/Linux_2k.log",
        "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}"],
        "st": 0.65,
        "splitters": r'(\s+|\(|\)|=)',
        "ngram_threshold": 0.2,
        "n": 3,
        "parsing_method": "combined",
    },
    "Android": {
        "log_file": "Android/Android_2k.log",
        "log_format": "<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>",
        "regex": [
            r"(/[\w-]+)+",
            r"([\w-]+\.){2,}[\w-]+",
            r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b",
        ],
        "st": 1,
        "splitters": r'(\s+|\(|\)|=)',
        "ngram_threshold": 0.4,
        "n": 3,
        "parsing_method": "combined",
    },
    "HealthApp": {
        "log_file": "HealthApp/HealthApp_2k.log",
        "log_format": "<Time>\|<Component>\|<Pid>\|<Content>",
        "regex": [],
        "st": 0.8,
        "splitters": r'(\s+|\[|\]|=)',
        "ngram_threshold": 0.8,
        "n": 3,
        "parsing_method": "combined",
    },
    "Apache": {
        "log_file": "Apache/Apache_2k.log",
        "log_format": "\[<Time>\] \[<Level>\] <Content>",
        "regex": [r"(\d+\.){3}\d+"],
        "st": 0.5,
        "splitters": r'(\s+)',
        "ngram_threshold": 0.8,
        "n": 3,
        "parsing_method": "seqDist",
    },
    "Proxifier": {
        "log_file": "Proxifier/Proxifier_2k.log",
        "log_format": "\[<Time>\] <Program> - <Content>",
        "regex": [
            r"([\w-]+\.)+[\w-]+(:\d+)?",
            r"\d{2}:\d{2}(:\d{2})*",
            r"[KGTM]B",
        ],
        "st": 0.8,
        "splitters": r'(\s+|\(|\))',
        "ngram_threshold": 0.7,
        "n": 3,
        "parsing_method": "ngram",
    },
    "OpenSSH": {
        "log_file": "OpenSSH/OpenSSH_2k.log",
        "log_format": "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+"],
        "st": 0.8,
        "splitters": r'(\s+|\(|\)|=|:)',
        "ngram_threshold": 0.6,
        "n": 3,
        "parsing_method": "combined",
    },
    "OpenStack": {
        "log_file": "OpenStack/OpenStack_2k.log",
        "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
        "regex": [
            r"((\d+\.){3}\d+,?)+",
            r"/.+?\s",
            r"\d+",
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        ],
        "st": 0.95,
        "splitters": r'(\s+|\[|\])',
        "ngram_threshold": 0.7,
        "n": 4,
        "parsing_method": "seqDist",
    },
    "Mac": {
        "log_file": "Mac/Mac_2k.log",
        "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
        "regex": [r"([\w-]+\.){2,}[\w-]+"],
        "st": 0.85,
        "splitters": r'(\s+|\[|\]|\(|\)|=)',
        "ngram_threshold": 0.8,
        "n": 3,
        "parsing_method": "seqDist",
    },
}

benchmark_result = []
for dataset, setting in benchmark_settings.items():
    print("\n=== Evaluation on %s ===" % dataset)
    indir = os.path.join(input_dir, os.path.dirname(setting["log_file"]))
    log_file = os.path.basename(setting["log_file"])

    parser = LogParser(
        log_format=setting['log_format'],
        indir=indir,
        outdir=output_dir,
        st=setting['st'],
        rex=setting["regex"],
        splitters=setting.get("splitters", None),
        ngram_threshold=setting['ngram_threshold'],
        n=setting['n'],
        dataset_name=dataset,
        parsing_method=setting['parsing_method']
    )
    start_time = time.time()
    parser.parse(log_file)
    parsing_time = time.time() - start_time
    parsing_time = round(parsing_time, 3)
    GA, FGA, FTA, PTA, RTA = evaluator.evaluate(
        groundtruth=os.path.join(indir, log_file + "_structured_rev.csv"),
        # groundtruth=os.path.join(indir, log_file + "_structured_corrected.csv"),
        parsedresult=os.path.join(output_dir, log_file + "_structured.csv"),
    )

    GA = round(GA, 3)
    FGA = round(FGA, 3)
    PTA = round(PTA, 3)
    RTA = round(RTA, 3)
    FTA = round(FTA, 3)

    benchmark_result.append([dataset, GA, FGA, FTA, PTA, RTA, parsing_time])

print("=== Overall evaluation results ===")
df_result = pd.DataFrame(benchmark_result, columns=["Dataset", "GA", "FGA", "FTA", "PTA", "RTA", "P_Time"])
df_result.set_index("Dataset", inplace=True)

average_GA = round(df_result["GA"].mean(), 3)
average_FGA = round(df_result["FGA"].mean(), 3)
average_PTA = round(df_result["PTA"].mean(), 3)
average_RTA = round(df_result["RTA"].mean(), 3)
average_FTA = round(df_result["FTA"].mean(), 3)
average_parsing_time = round(df_result["P_Time"].mean(), 3)


df_result.loc["Average"] = [average_GA, average_FGA, average_FTA, average_PTA,
                            average_RTA, average_parsing_time]

print(df_result)
output_csv_file = os.path.join(output_dir, "Log_results.csv")
df_result.to_csv(output_csv_file)
print(f"Results have been saved to {output_csv_file}")
