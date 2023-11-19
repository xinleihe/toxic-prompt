import argparse
import torch
import json
import numpy
from openprompt.data_utils.data_processor import DataProcessor
from openprompt.data_utils.utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt import PromptForGeneration
from openprompt.utils.metrics import generation_metric
from openprompt import PromptDataLoader
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import numpy as np
import time

parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
# tested model are gpt2/t5
parser.add_argument("--model", type=str, default='t5')
parser.add_argument("--model_name_or_path", default='t5-base')
parser.add_argument("--dataset", type=str, default='TSD')
parser.add_argument("--epoch", type=int, default=5)
args = parser.parse_args()
print(args)


class MyDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["No", "Yes"]

    def random_sampling(self, dataset, sample_num):
        import random
        random.seed(42)
        # if exceed training data size, set to training data size, then;
        sample_num = min(sample_num, len(dataset["original_text"]))
        index_list = list(range(len(dataset["original_text"])))
        random.shuffle(index_list)
        selected_index_list = index_list[:sample_num]
        new_dataset = {
            "id": [
                dataset["id"][i] for i in selected_index_list], "original_text": [
                dataset["original_text"][i] for i in selected_index_list], "new_text": [
                dataset["new_text"][i] for i in selected_index_list], "label": [
                    dataset["label"][i] for i in selected_index_list]}
        return new_dataset

    def get_examples(self, data_dir, split):
        if split == "valid" or split == "dev":
            split = "validation"
        self.split = split

        dataset = json.loads(open(data_dir).read())
        dataset = dataset[split]
        # sample_num = 1000
        sample_num = -1
        if sample_num != -1 and split == "train":
            dataset = self.random_sampling(dataset, sample_num)
            print("%s, sample %d data." % (split, len(dataset["id"])))
        return self.transform(dataset)

    def transform(self, dataset):
        res = []
        # score_list = []
        for i in range(len(dataset["id"])):
            text_a = dataset['text'][i]
            tgt = dataset['new_text'][i]
            # label = dataset['label'][i]
            guid = "{}".format(dataset['id'][i])
            # res.append(InputExample(guid=guid, text_a=text_a, tgt_text=tgt, label=label))
            res.append(InputExample(guid=guid, text_a=text_a, tgt_text=tgt))
        original_avg = np.mean([row["toxicity"]
                               for row in dataset["perspective_score"]])
        new_avg = np.mean([row["toxicity"]
                          for row in dataset["new_perspective_score"]])
        print("%d samples in %s, original score: %.3f, new score: %.3f" %
              (len(res), self.split, original_avg, new_avg))
        return res


dataset = {}


select_dataset = args.dataset
# select_dataset = "TSD"
# select_dataset = "Parallel"
# select_dataset = "ParaDetox"
data_dir = "parsed_dataset/%s_perspective.json" % (select_dataset)
Processor = MyDataProcessor
dataset['train'] = Processor().get_train_examples(data_dir)
dataset['test'] = Processor().get_test_examples(data_dir)
class_labels = Processor().get_labels()

# load a pretrained model, its tokenizer, its config, and its
# TokenzerWrapper by one function
plm, tokenizer, model_config, WrapperClass = load_plm(
    args.model, args.model_name_or_path)
# Instantiating the PrefixTuning Template
mytemplate = PrefixTuningTemplate(
    model=plm,
    tokenizer=tokenizer,
    text='{"placeholder":"text_a"} {"special": "<eos>"} Generate a less toxic sentence:\n{"mask"}.',
    using_decoder_past_key_values=False)


train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
                                    # be sure to pass predict_eos_token=True if
                                    # your template doesn't contain one, or you
                                    # model may fail to stop generation.
                                    batch_size=5, shuffle=True, teacher_forcing=True, predict_eos_token=True,
                                    truncate_method="head")

test_dataloader = PromptDataLoader(
    dataset=dataset["test"],
    template=mytemplate,
    tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=256,
    decoder_max_length=256,
    batch_size=5,
    shuffle=False,
    teacher_forcing=False,
    predict_eos_token=True,
    truncate_method="head")

# load the pipeline model PromptForGeneration.

use_cuda = True
prompt_model = PromptForGeneration(
    plm=plm,
    template=mytemplate,
    freeze_plm=True,
    tokenizer=tokenizer,
    plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()


# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training.
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in mytemplate.named_parameters() if (
                not any(
                    nd in n for nd in no_decay)) and p.requires_grad], "weight_decay": 0.0, }, {
                        "params": [
                            p for n, p in mytemplate.named_parameters() if any(
                                nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0, }, ]


optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
tot_step = len(train_dataloader) * args.epoch
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)


# Define evaluate function
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(
            inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(
        generated_sentence,
        groundtruth_sentence,
        "sentence_bleu")
    print("test_score", score, flush=True)
    return generated_sentence, groundtruth_sentence


generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": [[628], [198]]
}

# training and generation.
global_step = 0
tot_loss = 0
log_loss = 0
t_start = time.time()
for epoch in range(args.epoch):
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        global_step += 1
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if global_step % 100 == 0:
            print(
                "Epoch {}, global_step {} average loss: {} lr: {}".format(
                    epoch,
                    global_step,
                    (tot_loss - log_loss) / 500,
                    scheduler.get_last_lr()[0]),
                flush=True)
            log_loss = tot_loss
    print(time.time() - t_start)


generated_sentence, groundtruth_sentence = evaluate(
    prompt_model, test_dataloader)
original_sentence = [json.loads((dataset['test'][i].to_json_string()))[
    'text_a'] for i in range(len(dataset['test']))]

# get span from the original testing dataset, note that for task 3, the
# span is empty
f = json.loads(open(data_dir).read())
original_span = f["test"]['label']

with open(f"sfs_out/task23/{select_dataset}_{args.model_name_or_path}_{args.plm_eval_mode}.txt", 'w') as f:
    for i in range(len(generated_sentence)):
        ret = {
            "original": original_sentence[i],
            "original_span": original_span[i],
            "ground_truth": groundtruth_sentence[i],
            "generated": generated_sentence[i]}
        f.write(json.dumps(ret) + "\n")
