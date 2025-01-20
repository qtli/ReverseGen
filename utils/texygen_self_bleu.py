import json
import os
import pdb
from multiprocessing import Pool
import random
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from sacrebleu.metrics import BLEU, CHRF, TER
import numpy as np
from abc import abstractmethod
# from curiosity_self_bleu import get_attack_samples
# from common import remove_repeat
bleu = BLEU()


# based on: https://github.com/geek-ai/Texygen/blob/3104e22ac75f3cc2070da2bf5e2da6d2bef149ad/utils/metrics/Bleu.py#L10
# https://www.nltk.org/_modules/nltk/translate/bleu_score.html
class Metrics:
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass


class SelfBleu(Metrics):
    def __init__(self, test_data: list, gram=3, hypo_size=-1, ref_size=-1):
        super().__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_data
        self.gram = gram
        self.hypo_size = hypo_size
        self.ref_size = ref_size
        self.is_first = True
        self.reference = None

    def get_name(self):
        return self.name


    def get_reference(self):
        if self.reference is None:
            reference = list()
            for text in self.test_data:
                text = nltk.word_tokenize(text)
                reference.append(text)
            # reference = self.test_data
            self.reference = reference
            return reference
        else:
            return self.reference


    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()  # 分词放进 reference list
            self.is_first = False

        if is_fast:
            reference = self.get_reference()
            if self.hypo_size > 0:
                # 只采样部分样本计算bleu
                # random.shuffle(reference)
                # reference = reference[0:self.sample_size]
                reference = random.sample(reference, k=self.hypo_size)
            return self.get_bleu_parallel(reference=reference)

        return self.get_bleu_parallel()


    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.test_data
        weight = tuple((1. / ngram for _ in range(ngram)))
        print("weight: ", weight)
        # with open(self.test_data) as test_data:
        for hypothesis in self.test_data:
            hypothesis = nltk.word_tokenize(hypothesis)
            bleu.append(nltk.translate.bleu_score.sentence_bleu(
                reference,
                hypothesis,
                weight,
                smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        bleu = nltk.translate.bleu_score.sentence_bleu(
            reference,   # list of token list
            hypothesis,  # token list
            weight,
            smoothing_function=SmoothingFunction().method1)  # method1, Laplace平滑
        if isinstance(weight, list):
            bleu = np.asarray(bleu).mean(axis=0)


        each_scores = []
        for ref in reference:
            single_bleu = nltk.translate.bleu_score.sentence_bleu(
                [ref],  # list of token list
                hypothesis,  # token list
                weight,
                smoothing_function=SmoothingFunction().method1)
            if isinstance(weight, list):
                single_bleu = np.asarray(single_bleu).mean(axis=0)
            each_scores.append(single_bleu)


        ref_score = zip(reference, each_scores)
        sorted_ref_score = sorted(ref_score, key=lambda x: x[1], reverse=True)
        sorted_ref, sorted_score = zip(*sorted_ref_score)

        return {
            "bleu": bleu,
            "sorted_references": sorted_ref,
            "sorted_scores": sorted_score
        }


    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()

        if isinstance(ngram, int):
            weight = tuple((1. / ngram for _ in range(ngram)))
        else:
            # weight = {f"{n}-gram": ([1. / n] * n) for n in ngram}
            weight = [tuple((1. / n for _ in range(n))) for n in ngram]


        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)

        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            if self.ref_size > 0:
                other = random.sample(other, k=self.ref_size)
            # pool.apply_async 异步非阻塞，不用等待当前进程执行完毕，随时跟进操作系统调度，进行进程切换，多个进程并行执行，提高程序执行效率
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))
        score = 0.0
        scores = []
        cnt = 0
        for i in result:
            score += i.get()["bleu"]
            scores.append(i.get()["bleu"])
            cnt += 1
        # print("score list: ", scores)
        pool.close()
        pool.join()

        return score / cnt



class NormalBleu(Metrics):
    def __init__(self, reference_data: list, hypothesis_data: list, gram=3, hypo_size=-1, ref_size=-1):
        super().__init__()
        self.name = 'Normal-Bleu'
        self.reference_data = reference_data
        self.hypothesis_data = hypothesis_data

        self.gram = gram
        self.hypo_size = hypo_size
        self.ref_size = ref_size
        self.is_first = True
        self.reference = None

    def get_name(self):
        return self.name


    def get_reference(self):
        if self.reference is None:
            reference = list()
            for text in self.reference_data:
                text = nltk.word_tokenize(text)
                reference.append(text)
            self.reference = reference

        return self.reference

    def get_hypothesis(self):
        if self.hypothesis is None:
            hypothesis = list()
            for text in self.hypothesis_data:
                text = nltk.word_tokenize(text)
                hypothesis.append(text)
            self.hypothesis = hypothesis

        return self.hypothesis

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()  # 分词放进 reference list
            self.get_hypothesis()
            self.is_first = False

        if is_fast:
            reference = self.get_reference()
            hypothesis = self.get_hypothesis()
            if self.hypo_size > 0:
                # 只采样部分样本计算bleu
                # random.shuffle(reference)
                # reference = reference[0:self.sample_size]
                reference = random.sample(reference, k=self.hypo_size)
            return self.get_bleu_parallel(reference=reference, hypothesis=hypothesis)

        return self.get_bleu_parallel(reference=self.reference, hypothesis=self.hypothesis)


    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.test_data
        weight = tuple((1. / ngram for _ in range(ngram)))
        print("weight: ", weight)
        # with open(self.test_data) as test_data:
        for hypothesis in self.test_data:
            hypothesis = nltk.word_tokenize(hypothesis)
            bleu.append(nltk.translate.bleu_score.sentence_bleu(
                reference,
                hypothesis,
                weight,
                smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        bleu = nltk.translate.bleu_score.sentence_bleu(
            reference,   # list of token list
            hypothesis,  # token list
            weight,
            smoothing_function=SmoothingFunction().method1)  # method1, Laplace平滑
        if isinstance(weight, list):
            bleu = np.asarray(bleu).mean(axis=0)


        each_scores = []
        for ref in reference:
            single_bleu = nltk.translate.bleu_score.sentence_bleu(
                [ref],  # list of token list
                hypothesis,  # token list
                weight,
                smoothing_function=SmoothingFunction().method1)
            if isinstance(weight, list):
                single_bleu = np.asarray(single_bleu).mean(axis=0)
            each_scores.append(single_bleu)


        ref_score = zip(reference, each_scores)
        sorted_ref_score = sorted(ref_score, key=lambda x: x[1], reverse=True)
        sorted_ref, sorted_score = zip(*sorted_ref_score)

        return {
            "bleu": bleu,
            "sorted_references": sorted_ref,
            "sorted_scores": sorted_score
        }


    def get_bleu_parallel(self, reference=None, hypothesis=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        if hypothesis is None:
            hypothesis = self.get_hypothesis()

        if isinstance(ngram, int):
            weight = tuple((1. / ngram for _ in range(ngram)))
        else:
            # weight = {f"{n}-gram": ([1. / n] * n) for n in ngram}
            weight = [tuple((1. / n for _ in range(n))) for n in ngram]


        pool = Pool(os.cpu_count())
        result = list()
        reference_num = len(reference)
        hypothesis_num = len(hypothesis)

        for index in range(hypothesis_num):
            # todo: mainly revised part

            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            if self.ref_size > 0:
                other = random.sample(other, k=self.ref_size)
            # pool.apply_async 异步非阻塞，不用等待当前进程执行完毕，随时跟进操作系统调度，进行进程切换，多个进程并行执行，提高程序执行效率
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))
        score = 0.0
        scores = []
        cnt = 0
        for i in result:
            score += i.get()["bleu"]
            scores.append(i.get()["bleu"])
            cnt += 1


        pool.close()
        pool.join()

        return score / cnt


if __name__ == '__main__':
    # https://github.com/geek-ai/Texygen
    # all_str_samples, all_str_prompts, all_str_outputs, all_str_targets = (
    #     get_attack_samples(filename="../samples/llama7b_hh_query_dpo_top_0.9_sample_times_1.json"))
    all_str_outputs = ["How are you?", "The flower is so beautiful", "are you hungry?"]
    # all_str_outputs = json.load(open("../explore_dataset/MaliciousInstruct.json"))
    # all_str_outputs = remove_repeat(all_str_outputs)
    # all_str_targets = remove_repeat(all_str_targets)
    # all_str_outputs = all_str_targets

    # all_str_outputs = all_str_outputs[:10]
    # print("samples: ", '\n'.join(all_str_outputs[:3]))
    # all_str_outputs = json.load(open("../explore_dataset/MaliciousInstruct.json"))
    # data = json.load(open("../explore_dataset/eval_sft.json"))
    # all_str_outputs = []
    # for item in data:
    #     all_str_outputs.append(item["target"])
    # print("size of outputs: ", len(all_str_outputs))


    self_bleu_module = SelfBleu(
        test_data=all_str_outputs,
        gram=4,
        hypo_size=-1,
        ref_size=-1
    )
    score = self_bleu_module.get_score()
    print("score: ", score)


