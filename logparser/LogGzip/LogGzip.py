
from datetime import datetime
from collections import Counter
import os
import pandas as pd
import regex as re
import operator
import random
from collections import defaultdict
from typing import Any, Callable, Optional
import numpy as np

from logparser.LogGzip.compressors import DefaultCompressor
from tqdm import tqdm
from logparser.LogGzip.utils import *
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

RED = "\033[31m"
RESET = "\033[0m"
PINK = "\033[38;2;255;192;203m"


def NCD(c1: float, c2: float, c12: float) -> float:

    distance = (c12 - min(c1, c2)) / max(c1, c2)
    return distance


def agg_by_concat_space(t1: str, t2: str) -> str:

    return t1 + " " + t2

class LogParser:
    def __init__(
        self,
        logname,
        log_format,
        compressor: DefaultCompressor,
        agg_by_concat_space=agg_by_concat_space,
        NCD=NCD,
        indir="./",
        outdir="./result/",
        threshold=2,
        delimeter=[],
        rex=[],
    ):
        self.logformat = log_format
        self.path = indir
        self.savePath = outdir
        self.rex = rex
        self.df_log = None
        self.logname = logname
        self.agg_by_concat_space = agg_by_concat_space
        self.NCD=NCD
        self.compressor = compressor
        self.distance_matrix: list = []
        self.delimeter = delimeter
        self.sentence_clusters = {}

    def parse(self, logName):
        print("Parsing file: " + os.path.join(self.path, logName))
        starttime = datetime.now()
        self.logName = logName

        self.load_data()

        sentences = self.df_log["Content"].tolist()

        # Classify based on message length
        length_clusters = self.classify_by_length(sentences)

        # Perform clustering for each length cluster using compressed distances and DBSCAN
        for length, sentences_in_length_cluster in length_clusters.items():
            print(f"Clustering for length {length}...")

            if len(sentences_in_length_cluster) > 1:  # Ensure there are at least 2 sentences to compare
                # Calculate distance matrix for sentences in the current length cluster
                distance_matrix = self.calc_dis(sentences_in_length_cluster)
                distance_matrix_array = np.array(distance_matrix)

                # Use DBSCAN for clustering based on the distance matrix
                db = DBSCAN(eps=0.1, min_samples=2, metric='precomputed')
                clusters = db.fit_predict(distance_matrix_array)

                # Assign each sentence in the current length cluster with its cluster ID
                for sentence, cluster_id in zip(sentences_in_length_cluster, clusters):
                    self.sentence_clusters[sentence] = f"{length}_{cluster_id}"
            else:
                # If only one sentence in this length cluster, it forms a cluster by itself.
                self.sentence_clusters[sentences_in_length_cluster[0]] = f"{length}_0"

        # Generate the results
        self.generateresult(sentences)

    def classify_by_length(self, sentences):
        length_clusters = defaultdict(list)

        # Classify sentences into length clusters
        for sentence in sentences:
            length = len(sentence)
            length_clusters[length].append(sentence)

        return length_clusters

    def calc_dis(self, data: list) -> np.ndarray:
        num_sentences = len(data)
        distance_matrix = np.zeros((num_sentences, num_sentences))

        for i, t1 in tqdm(enumerate(data)):
            t1_compressed = self.compressor.get_compressed_len(t1)
            for j, t2 in enumerate(data):
                t2_compressed = self.compressor.get_compressed_len(t2)
                t1t2_compressed = self.compressor.get_compressed_len(self.agg_by_concat_space(t1, t2))
                distance = self.NCD(t1_compressed, t2_compressed, t1t2_compressed)
                distance_matrix[i, j] = distance

        return distance_matrix

    def get_template(self, preprocessed_sentences):
        # Initialize a Counter for each word position in the sentences
        word_positions = defaultdict(Counter)
        for sentence in preprocessed_sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                word_positions[i].update([word])

        # Generate template by choosing the most common word at each position
        template = " ".join(word_counts.most_common(1)[0][0] for word_counts in word_positions.values())
        return template

    def generateresult(self, sentences):
        # Initialize the DataFrame to hold the template results
        df_event = pd.DataFrame(columns=["EventId", "EventTemplate", "Occurrences"])

        # Generate templates based on clusters
        for event_id, cluster_sentences in self.sentence_clusters.items():
            if cluster_sentences:
                # Preprocess sentences to generalize them
                preprocessed_sentences = [self.preprocess(sentence) for sentence in cluster_sentences]

                # Extract template
                template = self.get_template(preprocessed_sentences)
                occurrences = len(cluster_sentences)

                # Create a new DataFrame row
                new_row = pd.DataFrame([[event_id, template, occurrences]],
                                       columns=["EventId", "EventTemplate", "Occurrences"])

                # Add new row to the DataFrame
                df_event = pd.concat([df_event, new_row], ignore_index=True)

        # Ensure EventId columns are of the same data type
        self.df_log['EventId'] = self.df_log['EventId'].astype(str)
        df_event['EventId'] = df_event['EventId'].astype(str)

        # Add 'EventId' and 'EventTemplate' to the log dataframe
        self.df_log = self.df_log.merge(df_event, on="EventId", how="left")

        # Fill NaNs with "No template" for rows that didn't find a match in df_event
        self.df_log["EventTemplate"].fillna("No template", inplace=True)

        # Save the final DataFrame to a CSV file
        self.df_log.to_csv(os.path.join(self.savePath, self.logName + "_structured.csv"), index=False)

        # Print summary
        print(f"Total clusters: {len(df_event)}")
        print(f"Result saved to: {self.savePath}")

    #这是一个预处理方法，接收一个参数line。方法通过遍历self.rex并对line进行预处理，将每个找到的模式替换为"<*>"。处理完成后的line作为结果返回。
    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, "<*>", line)
        return line

    #使用generate_logformat_regex方法生成用于分析日志文件的正则表达式regex和标签列表headers。
    # 然后，它使用log_to_dataframe方法将日志文件转换成一个pandas的DataFrame。
    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.logformat)
        self.df_log = self.log_to_dataframe(
            os.path.join(self.path, self.logName), regex, headers, self.logformat
        )
        self.df_log["EventId"] = -1


    #接收一个logformat（日志格式）作为参数，返回一个用于分割日志信息的正则表达式以及相应的标题列表。函数通过拆分logformat参数并构建正则表达式模式完成此任务。
    def generate_logformat_regex(self, logformat):
        """
        Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r"(<[^<>]+>)", logformat)
        regex = ""
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(" +", "\\\\s+", splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip("<").strip(">")
                regex += "(?P<%s>.*?)" % header
                headers.append(header)
        regex = re.compile("^" + regex + "$")
        return headers, regex

    #通用模块
    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """Function to transform log file to dataframe"""
        log_messages = []
        linecount = 0
        with open(log_file, "r") as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, "LineId", None)
        logdf["LineId"] = [i + 1 for i in range(linecount)]
        return logdf
def save_result(dataset, df_output, template_set):
    df_output.to_csv("Parseresult/" + dataset + "result.csv", index=False)
    with open("Parseresult/" + dataset + "_template.csv", "w") as f:
        for k1 in template_set.keys():
            f.write(" ".join(list(k1)))
            f.write("  " + str(len(template_set[k1])))
            f.write("\n")
        f.close()

