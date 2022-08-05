import logging
import argparse
import datasets
import torch
import sacremoses
from tqdm.auto import tqdm
from functools import partial
from transformers import (
	set_seed,
	BertTokenizer,
	BartTokenizer,
	BartForConditionalGeneration,
	AutoConfig,
	AutoTokenizer,
	AutoModelForMaskedLM,
)
from DataCollatorForNoiser import DataCollatorForNoiser
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_file", type=str, default=None, help="A text file containing the inference data.")
	parser.add_argument("--input_file_b", type=str, default=None, help="A text file containing the inference data.")
	parser.add_argument("--ratio_file", type=str, default=None, help="A text file containing the inference data.")
	parser.add_argument("--noised_file", type=str, default=None, help="Where to output the noised results.")
	parser.add_argument("--output_file", type=str, default=None, help="Where to output the mlm results.")
	parser.add_argument("--model_path", type=str, help="Path to pretrained model", required=True)
	parser.add_argument("--model_type", type=str, choices=("cBERT-HIT", "cBERT", "cBART", "BERT", "BART"))
	parser.add_argument("--batch_size", type=int, default=8, help="Batch size (per device) for inference.")	
	parser.add_argument("--seed", type=int, default=None, help="A seed for reproduciblility.")
	parser.add_argument("--num_beams", type=int, default=4, help="Only for BART generation.")
	parser.add_argument("--max_length", type=int, default=256)

	parser.add_argument("--mask_strategy", type=str, default="mask_to_multi", choices=("mask_to_multi", "mask_to_single"))
	parser.add_argument("--mask_length", type=str, default="subword", choices=("subword", "wwm", "subspan", "wwmspan"))
	parser.add_argument("--poisson_lambda", type=float, default=2)
	parser.add_argument("--mask_ratio", type=float, default=0.3)
	parser.add_argument("--inde_ratio", type=float, default=0.1)
	parser.add_argument("--random_ratio", type=float, default=0.5)
	parser.add_argument("--vocab_lower_bound", type=int, default=106)
	# Used as the lower_bound when performing random substitution. Default is 106 for BERT.
	parser.add_argument("--negative_time", type=int, default=30)
	args = parser.parse_args()

	print(args)

	# If passed along, set the training seed now.
	if args.seed is not None:
		set_seed(args.seed)

	if args.input_file_b is not None:
		if args.ratio_file is None:
			infer_dataset = datasets.Dataset.from_dict({"src":[line.strip() for line in open(args.input_file, encoding='utf-8').readlines()],
														"tgt":[line.strip() for line in open(args.input_file_b, encoding='utf-8').readlines()]})
		else:
			infer_dataset = datasets.Dataset.from_dict({"src":[line.strip() for line in open(args.input_file, encoding='utf-8').readlines()],
														"tgt":[line.strip() for line in open(args.input_file_b, encoding='utf-8').readlines()],
														"ratio":[float(line.strip()) for line in open(args.ratio_file, encoding='utf-8').readlines()]})			
	else:
		if args.ratio_file is None:
			infer_dataset = datasets.Dataset.from_dict({"tgt":[line.strip() for line in open(args.input_file, encoding='utf-8').readlines()]})
		else:
			infer_dataset = datasets.Dataset.from_dict({"tgt":[line.strip() for line in open(args.input_file, encoding='utf-8').readlines()],
														"ratio":[float(line.strip()) for line in open(args.ratio_file, encoding='utf-8').readlines()]})

	if args.model_type == "cBART":
		tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
		model = BartForConditionalGeneration.from_pretrained(args.model_path, local_files_only=True).cuda()
	elif args.model_type == "BART":
		tokenizer = BartTokenizer.from_pretrained(args.model_path, local_files_only=True)
		model = BartForConditionalGeneration.from_pretrained(args.model_path, local_files_only=True).cuda()
	else:
		config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
		tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
		model = AutoModelForMaskedLM.from_pretrained(args.model_path,
													 config=config,
													 local_files_only=True).cuda()

	if args.model_type == "cBERT-HIT":
		from ltp import LTP
		from run_chinese_ref import prepare_ref
		ltp = LTP()

	def tokenize_function(examples):

		if args.model_type == "cBERT-HIT":
			ref_ids = prepare_ref(examples["tgt"], ltp, tokenizer)

		if args.input_file_b is not None:
			features = tokenizer(
				examples["src"],
				examples["tgt"],
				truncation=True,
				max_length=args.max_length,
				# We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
				# receives the `special_tokens_mask`.
				return_special_tokens_mask=True,
			)
		else:
			features = tokenizer(
				examples["tgt"],
				truncation=True,
				max_length=args.max_length,
				# We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
				# receives the `special_tokens_mask`.
				return_special_tokens_mask=True,
			)			

		if args.model_type == "cBERT-HIT":
			features["chinese_ref"] = ref_ids

		if args.ratio_file is not None:
			features["ratio"] = examples["ratio"]

		return features


	infer_dataset = infer_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)

	md = sacremoses.MosesDetokenizer()

	if args.negative_time > 1:
		all_noised_lines = []
		all_output_lines = []

	for n in range(args.negative_time):
		data_collator = DataCollatorForNoiser(tokenizer=tokenizer, args=args)
		infer_dataloader = torch.utils.data.DataLoader(infer_dataset, shuffle=False, collate_fn=partial(data_collator, args=args), batch_size=args.batch_size)

		noised_lines = []
		output_lines = []

		if args.model_type in ["cBART", "BART"]:
			for batch in tqdm(infer_dataloader, desc="Inferring "+str(n)+" group"):
				with torch.no_grad():
					output = model.generate(
						batch["input_ids"].cuda(),
						attention_mask=batch["attention_mask"].cuda(),
						max_length=args.max_length,
						num_beams=args.num_beams,
					)
					output_tokens = tokenizer.batch_decode(output, skip_special_tokens=True)
					for j in range(len(batch["input_ids"])):
						noised_sent = torch.masked_select(batch["input_ids"][j], batch["attention_mask"][j]==1)
						noised_sent = tokenizer.decode(noised_sent[1:-1])
						noised_lines.append(md.detokenize(noised_sent.split()))
					for output_sent in output_tokens:
						output_lines.append(md.detokenize(output_sent.split()))

		else:
			for batch in tqdm(infer_dataloader, desc="Inferring "+str(n)+" group"):

				ori_batch_ids = batch["input_ids"].clone()

				with torch.no_grad():
					if "token_type_ids" in batch.keys():
						output = model(input_ids=batch['input_ids'].cuda(),
									   attention_mask=batch['attention_mask'].cuda(),
									   token_type_ids=batch['token_type_ids'].cuda())
					else:
						output = model(input_ids=batch['input_ids'].cuda(),
									   attention_mask=batch['attention_mask'].cuda())

					# output_indices = output.logits.max(dim=-1).indices
					# for j in range(len(batch['input_ids'])):
					# 	print(tokenizer.decode(batch['input_ids'][j]))
					# 	print(tokenizer.decode(output_indices[j]))

					output_prob = torch.nn.functional.softmax(output.logits, -1).topk(5, dim=-1, largest=True, sorted=True).indices
					# [batch_size, seq_len, k]
					mask_index_list = ((batch['input_ids'] != tokenizer.cls_token_id) &\
									   (batch['input_ids'] != tokenizer.sep_token_id) &\
									   (batch['input_ids'] != tokenizer.pad_token_id)).nonzero().tolist()
					for j in range(len(batch['input_ids'])):
						sent = tokenizer.decode(batch['input_ids'][j])
						if tokenizer.pad_token in sent: # sent_pair
							pad_id = sent.index(tokenizer.pad_token)
							sent = sent[:pad_id]
						noised_lines.append(md.detokenize(sent.split()))

					for mask_index in mask_index_list:
						x, y = mask_index[0], mask_index[1]
						# x 是这个 batch 的句子的 id，y 是这个句子的 token 的 id
						replacement = output_prob[x][y][0]
						topk_id = 0
						unacceptable_subs = [tokenizer.cls_token_id,
											 tokenizer.sep_token_id,
											 tokenizer.pad_token_id,
											 tokenizer.unk_token_id]
						while(replacement in unacceptable_subs):
							topk_id += 1
							replacement = output_prob[x][y][topk_id]
						batch['input_ids'][x][y] = replacement # 最后一位下标是topk的值
						
					for j in range(len(batch['input_ids'])):
						sent = tokenizer.decode(batch['input_ids'][j][1:])
						sep_id = sent.index(tokenizer.sep_token)
						if "token_type_ids" in batch.keys(): # sent_pair
							sent = sent[sep_id+5:]
							sep_id = sent.index(tokenizer.sep_token)
						output_lines.append(md.detokenize(sent[:sep_id].split()))

						if sent[:sep_id].strip() == "":
							raise Expectation("Generating empty sequence! Please check your codes.")

		if args.negative_time > 1:
			all_noised_lines.append(noised_lines)
			all_output_lines.append(output_lines)

	f_noised = open(args.noised_file, "w", encoding="utf-8")
	f_output = open(args.output_file, "w", encoding="utf-8")

	if args.negative_time == 1:
		for line in noised_lines:
			f_noised.write(line+"\n")
		for line in output_lines:
			f_output.write(line+"\n")
	else:
		for i in range(len(all_noised_lines[0])):
			temp_line = []
			for noised_lines in all_noised_lines:
				temp_line.append(noised_lines[i])
			f_noised.write(" ||| ".join(temp_line)+"\n")

		for i in range(len(all_output_lines[0])):
			temp_line = []
			for output_lines in all_output_lines:
				temp_line.append(output_lines[i])
			f_output.write(" ||| ".join(temp_line)+"\n")			