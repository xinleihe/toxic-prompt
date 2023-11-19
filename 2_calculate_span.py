import json
import copy
import re
import argparse

parser = argparse.ArgumentParser("")
parser.add_argument(
    "--file_path",
    type=str,
    default="sfs_out/task23/TSD_t5-small_True.txt")
args = parser.parse_args()

print(args)


def print_span(text, span):
    if span == []:
        print()
        return
    for i in range(len(span) - 1):
        print(text[span[i]], end='')
        if span[i + 1] - span[i] > 1:
            print(" ", end='')
    print(text[span[-1]])


def calculate_span(s):
    a = s["original"].lower()
    b = s['generated'].lower()
    span_ground_truth = s['original_span']

    s1 = re.findall(r"[\w'\"]+|[,.!?]", a)
    s2 = re.findall(r"[\w'\"]+|[,.!?]", b)

    not_finish_index = 0
    left_word = []
    for i in range(len(s1)):
        w1 = s1[i]

        match_flag = 0
        for j in range(not_finish_index, len(s2)):
            w2 = s2[j]
            if w1 == w2:
                match_flag = 1
                not_finish_index += 1
                break
        if match_flag != 1:
            left_word.append(w1)
        # print(i, not_finish_index, w1, w2, left_word)

    pred_span = []
    for w in left_word:
        start_index = a.find(w)
        end_index = start_index + len(w)
        pred_span += list(range(start_index, end_index))

    # if span_ground_truth == pred_span:
    #     return 1
    # else:
    #     return 0

    intersection = set(span_ground_truth) & set(pred_span)
    union = set(span_ground_truth) | set(pred_span)
    if len(span_ground_truth) == 0:
        iou = 1
    else:
        iou = len(intersection) / len(union)

    f1 = 0
    p = 0
    r = 0
    if len(span_ground_truth) == 0 and len(pred_span) == 0:
        f1 = 1
        p = 1
        r = 1
    elif len(span_ground_truth) == 0 and len(pred_span) != 0:
        f1 = 0
        p = 0
        r = 0
    elif len(span_ground_truth) != 0 and len(pred_span) == 0:
        f1 = 0
        p = 0
        r = 0
    else:
        p = len(set(span_ground_truth) & set(pred_span)) / len(pred_span)
        r = len(set(span_ground_truth) & set(
            pred_span)) / len(span_ground_truth)
        # print(p, r, span_ground_truth, pred_span)
        if p == 0 and r == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)

    return p, r, f1, iou


f = open(args.file_path).readlines()

cnt = 0
frac = 0
p_all = 0
r_all = 0
f1_all = 0
total = len(f)
for i in range(len(f)):
    p, r, f1, iou = calculate_span(json.loads(f[i]))
    p_all += p
    r_all += r
    f1_all += f1
    frac += iou
    if iou == 1:
        cnt += 1

print(
    cnt,
    total,
    cnt /
    total,
    frac /
    total,
    p_all /
    total,
    r_all /
    total,
    f1_all /
    total)
