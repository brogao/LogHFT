# =========================================================================
# Copyright (C) 2016-2023 LOGPAI (https://github.com/logpai).
# Copyright (C) 2023 gaiusyu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

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


    def parse(self, logName):
        print("Parsing file: " + os.path.join(self.path, logName))
        starttime = datetime.now()
        self.logName = logName

        self.load_data()

        sentences = self.df_log["Content"].tolist()


        distance_matrix = self.calc_dis(sentences)
        distance_matrix_array = np.array(distance_matrix)

        # Use KNN for classification
        neigh = NearestNeighbors(n_neighbors=3)  # Change the number of neighbors according to your need
        neigh.fit(distance_matrix_array)

        # Use DBSCAN for clustering
        db = DBSCAN(min_samples=5, metric='precomputed')  # Change this according to your need
        clusters = db.fit_predict(distance_matrix_array)

        # Associate each sentence with its cluster
        sentence_clusters = {sentence: cluster for sentence, cluster in zip(sentences, clusters)}

        # Store sentence clusters for later use or analysis
        template_set = sentence_clusters
        endtime = datetime.now()
        print("Parsing done...")
        print("Time taken   =   " + PINK + str(endtime - starttime) + RESET)

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.generateresult(template_set, sentences)



    def calc_dis(
            self, data: list, fast: bool = False
    ) -> None:


        for i, t1 in tqdm(enumerate(data)):
            distance4i = []
            if fast:
                t1_compressed = self.compressor.get_compressed_len_fast(t1)
            else:
                t1_compressed = self.compressor.get_compressed_len(t1)
            for j, t2 in enumerate(data):
                if fast:
                    t2_compressed = self.compressor.get_compressed_len_fast(t2)
                    t1t2_compressed = self.compressor.get_compressed_len_fast(self.agg_by_concat_space(t1, t2))
                else:
                    t2_compressed = self.compressor.get_compressed_len(t2)
                    t1t2_compressed = self.compressor.get_compressed_len(self.agg_by_concat_space(t1, t2))
                distance = self.NCD(
                    t1_compressed, t2_compressed, t1t2_compressed
                )
                distance4i.append(distance)
            self.distance_matrix.append(distance4i)
        return self.distance_matrix

    def generateresult(self, template_set, sentences):
        template_ = len(sentences) * [0]
        EventID = len(sentences) * [0]
        IDnumber = 0
        df_out = []
        for k1 in template_set.keys():
            df_out.append(["E" + str(IDnumber), k1, len(template_set[k1])])
            group_accuracy = {""}
            group_accuracy.remove("")
            for i in template_set[k1]:
                template_[i] = " ".join(k1)
                EventID[i] = "E" + str(IDnumber)
            IDnumber += 1

        self.df_log["EventId"] = EventID
        self.df_log["EventTemplate"] = template_
        self.df_log.to_csv(
            os.path.join(self.savePath, self.logName + "_structured.csv"), index=False
        )

        df_event = pd.DataFrame(
            df_out, columns=["EventId", "EventTemplate", "Occurrences"]
        )
        df_event.to_csv(
            os.path.join(self.savePath, self.logName + "_templates.csv"),
            index=False,
            columns=["EventId", "EventTemplate", "Occurrences"],
        )

    #计算距离
    # def calc_dis(
    #     self, data: list, train_data: Optional[list] = None, fast: bool = False
    # ) -> None:
    #     """
    #     计算“data”与其自身或“data”与“data”之间的距离
    #      `train_data` 并将距离附加到 `self.distance_matrix` 中。
    #
    #      论据：
    #          数据（列表）：用于计算之间距离的数据。
    #          train_data（列表）：[可选] 训练数据以计算与“data”的距离。
    #          fast (bool): [可选] 使用`self.compressor`的_fast压缩长度函数。
    #
    #     Returns:
    #         None: None
    #     """
    #
    #     data_to_compare = data
    #     if train_data is not None:
    #         data_to_compare = train_data
    #
    #     for i, t1 in tqdm(enumerate(data)):
    #         distance4i = []
    #         if fast:
    #             t1_compressed = self.compressor.get_compressed_len_fast(t1)
    #         else:
    #             t1_compressed = self.compressor.get_compressed_len(t1)
    #         for j, t2 in enumerate(data_to_compare):
    #             if fast:
    #                 t2_compressed = self.compressor.get_compressed_len_fast(t2)
    #                 t1t2_compressed = self.compressor.get_compressed_len_fast(
    #                     self.aggregation_func(t1, t2)
    #                 )
    #             else:
    #                 t2_compressed = self.compressor.get_compressed_len(t2)
    #                 t1t2_compressed = self.compressor.get_compressed_len(
    #                     self.aggregation_func(t1, t2)
    #                 )
    #             distance = self.distance_func(
    #                 t1_compressed, t2_compressed, t1t2_compressed
    #             )
    #             distance4i.append(distance)
    #         self.distance_matrix.append(distance4i)

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

    #接收一个logformat（日志格式）作为参数，返回一个用于分割日志信息的正则表达式以及相应的标题列表。函数通过拆分logformat参数并构建正则表达式模式完成此任务。
    def generate_logformat_regex(self, logformat):
        """Function to generate regular expression to split log messages"""
        headers = []
        splitters = re.split(r"(<[^<>]+>)", logformat)
        regex = ""
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(" +", "\\\s+", splitters[k])
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

# def output_result(parse_result):
#     template_set = {}
#     for key in parse_result.keys():
#         for pr in parse_result[key]:
#             sort = sorted(pr, key=lambda tup: tup[2])
#             i = 1
#             template = []
#             while i < len(sort):
#                 this = sort[i][1]
#                 if bool("<*>" in this):
#                     template.append("<*>")
#                     i += 1
#                     continue
#                 if exclude_digits(this):
#                     template.append("<*>")
#                     i += 1
#                     continue
#                 template.append(sort[i][1])
#                 i += 1
#             template = tuple(template)
#             template_set.setdefault(template, []).append(pr[len(pr) - 1][0])
#     return template_set


def save_result(dataset, df_output, template_set):
    df_output.to_csv("Parseresult/" + dataset + "result.csv", index=False)
    with open("Parseresult/" + dataset + "_template.csv", "w") as f:
        for k1 in template_set.keys():
            f.write(" ".join(list(k1)))
            f.write("  " + str(len(template_set[k1])))
            f.write("\n")
        f.close()

# #用来检查一个字符串中的数字占比是否大于或等于 30%。
# def exclude_digits(string):
#     """
#     exclude the digits-domain words from partial constant
#     """
#     pattern = r"\d"
#     digits = re.findall(pattern, string)
#     if len(digits) == 0:
#         return False
#     return len(digits) / len(string) >= 0.3



