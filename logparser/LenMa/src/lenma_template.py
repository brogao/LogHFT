import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from logparser.LenMa.src import template
from importlib import import_module



def NCD(c1: float, c2: float, c12: float) -> float:
    distance = (c12 - min(c1, c2)) / max(c1, c2)
    return distance
def agg_by_concat_space(t1: str, t2: str) -> str:
    return t1 + " " + t2


class LenmaTemplate(template.Template):
    def __init__(self, index=None, words=None, logid=None, json=None, compressor=None):  # Corrected parameter name
        if json is not None:
            # restore from the jsonized data.
            self._restore_from_json(json)
        else:
            # initialize with the specified index and words values.
            assert(index is not None)
            assert(words is not None)
            self._index = index
            self._words = words
            self._nwords = len(words)
            self._wordlens = [len(w) for w in words]
            self._counts = 1
            self._logid = [logid]
            self.compressor = compressor
            assert self.compressor is not None, "Compressor instance is None in LenmaTemplate"

    @property
    def wordlens(self):
        return self._wordlens

    def _dump_as_json(self):
        description = str(self)
        return json.dumps([self.index, self.words, self.nwords, self.wordlens, self.counts])

    def _restore_from_json(self, data):
        (self._index,
         self._words,
         self._nwords,
         self._wordlens,
         self._counts) = json.loads(data)

    def _try_update(self, new_words):
        try_update = [self.words[idx] if self._words[idx] == new_words[idx]
                      else '' for idx in range(self.nwords)]
        if (self.nwords - try_update.count('')) < 3:
            return False
        return True

    def _get_accuracy_score(self, new_words):
        # accuracy score
        # wildcard word matches any words
        fill_wildcard = [self.words[idx] if self.words[idx] != ''
                         else new_words[idx] for idx in range(self.nwords)]
        ac_score = accuracy_score(fill_wildcard, new_words)
        return ac_score

    def _get_wcr(self):
        return self.words.count('') / self.nwords

    def _get_accuracy_score2(self, new_words):
        wildcard_ratio = self._get_wcr()
        ac_score = accuracy_score(self.words, new_words)
        return (ac_score / (1 - wildcard_ratio), wildcard_ratio)

    def _get_similarity_score_cosine(self, new_words):
        wordlens = np.asarray(self._wordlens).reshape(1, -1)
        new_wordlens = np.asarray([len(w) for w in new_words]).reshape(1, -1)
        cos_score = cosine_similarity(wordlens, new_wordlens)
        return cos_score

    def __jaccarget_similarity_scored(self, new_words):
        ws = set(self.words) - set('')
        nws = set([new_words[idx] if self.words[idx] != '' else ''
                   for idx in range(len(new_words))]) - set('')
        return len(ws & nws) / len(ws | nws)

    def _count_same_word_positions(self, new_words):
        c = 0
        for idx in range(self.nwords):
            if self.words[idx] == new_words[idx]:
                c = c + 1
        return c

    def get_compression_distance(self, new_words):
        compressed_template = self.compressor.get_compressed_len(" ".join(self._words))
        compressed_new_words = self.compressor.get_compressed_len(" ".join(new_words))
        combined_text = agg_by_concat_space(" ".join(self._words), " ".join(new_words))
        compressed_combined = self.compressor.get_compressed_len(combined_text)

        distance = NCD(compressed_template, compressed_new_words, compressed_combined)
        return 1 - distance

    def get_similarity_score(self, new_words):
        if self._words[0] != new_words[0]:
            return 0

        ac_score = self._get_accuracy_score(new_words)
        if ac_score == 1:
            return 1

        comp_distance = self.get_compression_distance(new_words)
        return comp_distance

    def update(self, new_words, logid):
        self._counts += 1
        self._wordlens = [len(w) for w in new_words]
        self._words = [self.words[idx] if self._words[idx] == new_words[idx]
                       else '' for idx in range(self.nwords)]
        self._logid.append(logid)

    def print_wordlens(self):
        print('{index}({nwords})({counts}):{vectors}'.format(
            index=self.index,
            nwords=self.nwords,
            counts=self._counts,
            vectors=self._wordlens))

    def get_logids(self):
        return self._logid


class LenmaTemplateManager(template.TemplateManager):
    def __init__(self, threshold=0.9, predefined_templates=None, compressor_module=None, compressor_instance=None):
        super().__init__()
        self._threshold = threshold

        self.compressor_instance = compressor_instance

        # 确保compressor_instance在传递之前不是None
        assert compressor_instance is not None, "Compressor instance is None before passing to LenmaTemplateManager"


    def infer_template(self, words, logid):
        nwords = len(words)

        candidates = []
        for (index, template) in enumerate(self.templates):
            if nwords != template.nwords:
                continue
            score = template.get_similarity_score(words)
            if score < self._threshold:
                continue
            candidates.append((index, score))
        candidates.sort(key=lambda c: c[1], reverse=True)

        if len(candidates) > 0:
            index = candidates[0][0]
            self.templates[index].update(words, logid)
            return self.templates[index]

        # Pass compressor_instance instead of creating a new instance
        new_template = self._append_template(
            LenmaTemplate(index=len(self.templates), words=words, logid=logid, compressor=self.compressor_instance)
        )
        return new_template

    def dump_template(self, index):
        return self.templates[index]._dump_as_json()

    def restore_template(self, data):
        return LenmaTemplate(json=data)

# Assuming the following block is part of your main program
if __name__ == '__main__':
    import sys
    import datetime
    import importlib
    from templateminer.basic_line_parser import BasicLineParser as LP
    parser = LP()

    # 动态导入压缩器模块
    compressor_module_name = "compressor"  # 确保这是正确的模块名
    compressor_module = importlib.import_module(compressor_module_name)
    compressor_instance = compressor_module.Compressor()  # 创建compressor_instance
    assert compressor_instance is not None, "Failed to create compressor_instance."

    # 创建LenmaTemplateManager实例
    templ_mgr = LenmaTemplateManager(
        threshold=0.9,
        compressor_instance=compressor_instance  # 传递compressor_instance
    )

    nlines = 0
    line = sys.stdin.readline()
    while line:
        if nlines % 1000 == 0:
            print(f"{nlines} {datetime.datetime.now().timestamp()} {len(templ_mgr.templates)}")
        nlines += 1

        (month, day, timestr, host, words) = parser.parse(line)
        t = templ_mgr.infer_template(words)
        line = sys.stdin.readline()

    for t in templ_mgr.templates:
        print(t)

    for t in templ_mgr.templates:
        t.print_wordlens()