import regex as re
import os
import pandas as pd
import hashlib
from datetime import datetime
import nltk
from nltk import word_tokenize, pos_tag
import numpy as np
from functools import lru_cache
from difflib import SequenceMatcher
from collections import defaultdict


# download nltk (Just download once)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


class Logcluster:
    def __init__(self, logTemplate="", logIDL=None):
        self.logTemplate = logTemplate
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL


class LogParser:
    def __init__(
            self,
            log_format,
            indir="./",
            outdir="./result/",
            st=0.4,
            rex=[],
            splitters=r'(\s+)',
            ngram_threshold=0.8,
            keep_para=True,
            n=3,
            dataset_name="",
            parsing_method='combined'  # default method is combined
    ):
        self.path = indir
        self.st = st
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.splitters = splitters
        self.ngram_threshold = ngram_threshold
        self.n = n
        self.keep_para = keep_para
        self.dataset_name = dataset_name
        self.parsing_method = parsing_method
        self.parameter_cache = {}

    def preprocess(self, line, splitters):
        for currentRex in self.rex:
            line = re.sub(currentRex, "<*>", line)

        # Ensure splitters is a string or compiled pattern
        if not isinstance(splitters, (str, re.Pattern)):
            raise TypeError("splitters must be a string or compiled pattern")

        tokens = re.split(splitters, line)
        tokens = [token for token in tokens if token != '']

        processed_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] == '=' and i + 1 < len(tokens):
                if re.match(r'\w+', tokens[i + 1]):
                    processed_tokens.append(tokens[i])
                    processed_tokens.append('<*>')
                    i += 2
                    continue
            processed_tokens.append(tokens[i])
            i += 1

        for i, token in enumerate(processed_tokens):
            if token.lower() == "user" and i + 2 < len(processed_tokens):
                if not self.is_verb_adverb_preposition_or_symbol(processed_tokens[i + 2]):
                    processed_tokens[i + 2] = "<*>"
        for i in range(len(processed_tokens)):
            if self.dataset_name in ["OpenSSH", "Apache", "Zookeeper", "Mac", "HealthApp", "HPC", "BGL", "Proxifier", "Windows"]:
                if re.match(r'^\d', processed_tokens[i]) or re.match(r'\d$', processed_tokens[i]):
                    processed_tokens[i] = '<*>'
            if self.contains_special_characters(processed_tokens[i]):
                processed_tokens[i] = '<*>'

            # New rule: Convert specific characters or patterns to wildcard
            if any(char in processed_tokens[i] for char in ['/', '-']):
                processed_tokens[i] = '<*>'

        return processed_tokens

    def contains_special_characters(self, token):
        special_chars = set('/:@#_')
        char_count = defaultdict(int)

        for char in token:
            if char in special_chars:
                char_count[char] += 1

        if len(char_count) >= 2 or any(count >= 3 for count in char_count.values()):
            return True
        return False

    def is_verb_adverb_preposition_or_symbol(self, token):
        tokenized = word_tokenize(token)
        if not tokenized:
            return False
        pos_tagged = pos_tag(tokenized)
        return pos_tagged[0][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS', 'IN', '.', ',', ':', '(', ')', '[', ']', '{', '}', '!', '?', ' ']

    def log_to_dataframe(self, log_file, regex, headers, logformat):
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
                    print("[Warning] Skip line: " + line)
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, "LineId", None)
        logdf["LineId"] = [i + 1 for i in range(linecount)]
        print("Total lines: ", len(logdf))
        return logdf

    def generate_logformat_regex(self, logformat):
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

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []

        # 检查缓存
        if template_regex in self.parameter_cache:
            return self.parameter_cache[template_regex]

        template_regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template_regex)
        template_regex = re.sub(r"\\ +", r"\\s+", template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = (
            list(parameter_list)
            if isinstance(parameter_list, tuple)
            else [parameter_list]
        )

        # 存入缓存
        self.parameter_cache[template_regex] = parameter_list
        return parameter_list

    @lru_cache(maxsize=64)
    def seqDist(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == "<*>":
                numOfPar += 1
                continue
            if token1 == token2:
                simTokens += 1

        retVal = float(simTokens) / len(seq1)
        return retVal, numOfPar

    @lru_cache(maxsize=64)
    def ngram_similarity(self, seq1, seq2, n=3):
        def get_ngrams(seq, n):
            return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]

        ngrams1 = get_ngrams(seq1, n)
        ngrams2 = get_ngrams(seq2, n)
        if not ngrams1 or not ngrams2:
            return 0.0

        matcher = SequenceMatcher(None, ngrams1, ngrams2)
        return matcher.ratio()

    @lru_cache(maxsize=64)
    def jaccard_similarity(self, seq1, seq2):
        set1 = set(seq1)
        set2 = set(seq2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union if union != 0 else 0.0

    def getTemplate(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        retVal = []
        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                if not re.match(r'(\s+|\||\[|\]|\{|\}|\(|\))', word):  # 不替换分隔符
                    retVal.append("<*>")
                else:
                    retVal.append(word)
            i += 1
        return retVal

    def outputResult(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = []
        for logClust in logClustL:
            template_str = "".join(logClust.logTemplate)  # 使用原分隔符拼接模板
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]
            for logID in logClust.logIDL:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        df_event = pd.DataFrame(
            df_events, columns=["EventId", "EventTemplate", "Occurrences"]
        )
        self.df_log["EventId"] = log_templateids
        self.df_log["EventTemplate"] = log_templates
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(
                self.get_parameter_list, axis=1
            )
        self.df_log.to_csv(
            os.path.join(self.savePath, self.logName + "_structured.csv"), index=False
        )

        occ_dict = dict(self.df_log["EventTemplate"].value_counts())
        df_event = pd.DataFrame()
        df_event["EventTemplate"] = self.df_log["EventTemplate"].unique()
        df_event["EventId"] = df_event["EventTemplate"].map(
            lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8]
        )
        df_event["Occurrences"] = df_event["EventTemplate"].map(occ_dict)
        df_event.to_csv(
            os.path.join(self.savePath, self.logName + "_templates.csv"),
            index=False,
            columns=["EventId", "EventTemplate", "Occurrences"],
        )

    def parse(self, logName, splitters=None):
        print("Parsing file: " + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        logCluL = []

        self.load_data()

        count = 0
        length_dict = {}

        for idx, line in self.df_log.iterrows():
            log_messageL = self.preprocess(line["Content"], splitters or self.splitters)

            if len(log_messageL) not in length_dict:
                length_dict[len(log_messageL)] = []
            length_dict[len(log_messageL)].append(
                [log_messageL, idx + 1]
            )  # idx+1 indicating line number

        for length, loglist in length_dict.items():
            logCluL = self.process_loglist(logCluL, loglist, length)

        self.outputResult(logCluL)

        elapsed_time = datetime.now() - start_time
        print("Parsing done. [Time taken: {!s}]".format(elapsed_time))

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(
            os.path.join(self.path, self.logName), regex, headers, self.log_format
        )

    def process_loglist(self, logCluL, loglist, length):

        for log_messageL, line_index in loglist:
            match_cluster = None
            max_similarity = -1


            log_message_tuple = tuple(log_messageL)

            for logClust in logCluL:
                if len(logClust.logTemplate) != length:
                    continue

                logClust_tuple = tuple(logClust.logTemplate)

                if self.parsing_method in ['seqDist', 'combined']:
                    similarity, _ = self.seqDist(logClust_tuple, log_message_tuple)

                if self.parsing_method in ['ngram', 'combined']:
                    ngram_sim = self.ngram_similarity(logClust_tuple, log_message_tuple, self.n)
                    if self.parsing_method == 'ngram':
                        similarity = ngram_sim
                    elif self.parsing_method == 'combined':
                        similarity = max(similarity, ngram_sim)

                if self.parsing_method in ['jaccard', 'combined']:
                    jaccard_sim = self.jaccard_similarity(logClust_tuple, log_message_tuple)
                    if self.parsing_method == 'jaccard':
                        similarity = jaccard_sim
                    elif self.parsing_method == 'combined':
                        similarity = max(similarity, jaccard_sim)

                if similarity >= self.st and similarity > max_similarity:
                    max_similarity = similarity
                    match_cluster = logClust

            if match_cluster is None:
                new_cluster = Logcluster(logTemplate=log_messageL, logIDL=[line_index])
                logCluL.append(new_cluster)
            else:
                new_template = self.getTemplate(log_messageL, match_cluster.logTemplate)
                match_cluster.logIDL.append(line_index)
                if " ".join(new_template) != " ".join(match_cluster.logTemplate):
                    match_cluster.logTemplate = new_template

        return logCluL
