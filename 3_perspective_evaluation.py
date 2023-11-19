from perspective_api import PerspectiveApiScorer
from tqdm import tqdm
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser("")
parser.add_argument(
    "--file_path",
    type=str,
    default="sfs_out/task23/TSD_t5-small_True.txt")
parser.add_argument(
    "--key",
    type=str,
    default="your Perspective API KEY")
args = parser.parse_args()

print(args)


API_KEY = args.key
perspective_api_scorer = PerspectiveApiScorer(API_KEY)

f = open(args.file_path).readlines()


def get_perspective_score(s):
    text, perspective_api_scorer = s
    return perspective_api_scorer.get_scores(text)


def parse_list(text_list):
    res_all = []
    block_size = 50
    for i in tqdm(range(0, len(text_list), block_size)):
        start = i
        end = i + min(len(text_list) - start, block_size)
        texts = text_list[start:end]
        perspective_api_scorer = PerspectiveApiScorer(API_KEY)
        s = [[_, perspective_api_scorer] for _ in texts]
        with Pool(16) as p:
            res = p.map(get_perspective_score, s)
        res_all += res

    return res_all


score_list1 = []
score_list2 = []
t1 = []
t2 = []

for row in tqdm(f):
    row = json.loads(row)
    t1.append(row["original"])
    t2.append(row["generated"])

score_list1 = parse_list(t1)
score_list2 = parse_list(t2)

toxicity1 = np.array([row["toxicity"] for row in score_list1])
toxicity2 = np.array([row["toxicity"] for row in score_list2])

avg_score1 = np.mean(toxicity1)
avg_score2 = np.mean(toxicity2)
percentage_70_1 = np.sum(toxicity1 > 0.7) / len(toxicity1)
percentage_70_2 = np.sum(toxicity2 > 0.7) / len(toxicity2)

percentage_90_1 = np.sum(toxicity1 > 0.9) / len(toxicity1)
percentage_90_2 = np.sum(toxicity2 > 0.9) / len(toxicity2)
res1 = [avg_score1, percentage_70_1, percentage_90_1]
res2 = [avg_score2, percentage_70_2, percentage_90_2]
print(res1)
print(res2)
