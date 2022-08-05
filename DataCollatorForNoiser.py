import math
import torch
from transformers import BatchEncoding
import pdb


class DataCollatorForNoiser:
	def __init__(self, tokenizer, args):
		self.tokenizer = tokenizer
		self.mask_index = tokenizer.mask_token_id
		if args.model_type in ["cBERT", "cBART"]:
			import jieba_fast
			self.zh_tokenizer = jieba_fast.lcut
		self.mask_length = args.mask_length
		self.LOWER_BOUND = args.vocab_lower_bound
		self.UPPER_BOUND = tokenizer.vocab_size
		self.model_type = args.model_type

		self.mask_span_distribution = None
		
		if args.mask_length in ("subspan", "wwmspan"):
			_lambda = args.poisson_lambda # Poisson lambda

			lambda_to_the_k = 1
			e_to_the_minus_lambda = math.exp(-_lambda)
			k_factorial = 1
			ps = []
			for k in range(0, 128):
				ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
				lambda_to_the_k *= _lambda
				k_factorial *= k + 1
				if ps[-1] < 0.0000001:
					break
			ps = [0] + ps[1:]
			ps = torch.FloatTensor(ps)
			self.mask_span_distribution = torch.distributions.Categorical(ps)

	def word_starts(self, source, chinese_ref=None):
		if self.mask_length in ["subword", "subspan"]:
			is_word_start = torch.ones(source.size()).long()
			is_word_start[0] = 0
			is_word_start[-1] = 0

			return is_word_start

		if chinese_ref is not None:
			if chinese_ref != []:
				new_chinese_ref = []
				for r in chinese_ref:
					if r <= len(source):
						new_chinese_ref.append(r)
				chinese_ref = new_chinese_ref
				is_word_start = torch.ones(len(source)).long().\
									scatter(0, torch.LongTensor(chinese_ref), 0)
			else:
				is_word_start = torch.ones(source.size())
			is_word_start[0] = 0
			is_word_start[-1] = 0
			
			return is_word_start

		if args.model_type in ["cBERT", "cBART"]:
			raw_tokens = self.tokenizer.convert_ids_to_tokens(source)
			words = [raw_tokens[0]] + self.zh_tokenizer(''.join(raw_tokens[1:-1]), HMM=True) + [raw_tokens[-1]]

			def _is_chinese_char(c):
				"""Checks whether CP is the codepoint of a CJK character."""
				# This defines a "chinese character" as anything in the CJK Unicode block:
				#   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
				#
				# Note that the CJK Unicode block is NOT all Japanese and Korean characters,
				# despite its name. The modern Korean Hangul alphabet is a different block,
				# as is Japanese Hiragana and Katakana. Those alphabets are used to write
				# space-separated words, so they are not treated specially and handled
				# like the all of the other languages.
				if len(c) > 1:
					return all([_is_chinese_char(c_i) for c_i in c])
				cp = ord(c)
				if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
					(cp >= 0x3400 and cp <= 0x4DBF) or  #
					(cp >= 0x20000 and cp <= 0x2A6DF) or  #
					(cp >= 0x2A700 and cp <= 0x2B73F) or  #
					(cp >= 0x2B740 and cp <= 0x2B81F) or  #
					(cp >= 0x2B820 and cp <= 0x2CEAF) or
					(cp >= 0xF900 and cp <= 0xFAFF) or  #
						(cp >= 0x2F800 and cp <= 0x2FA1F)):  #
					return True

				return False

			def align_linear(atokens, btokens):
				a2c = []
				c2b = []
				a2b = []
				length = 0
				for tok in atokens:
					a2c.append([length + i for i in range(len(tok))])
					length += len(tok)
				for i, tok in enumerate(btokens):
					c2b.extend([i for _ in range(len(tok))])

				for i, amap in enumerate(a2c):
					bmap = [c2b[ci] for ci in amap]
					a2b.append(list(set(bmap)))
				return a2b
			
			raw_to_word_align = align_linear(raw_tokens, words)
			is_word_start = torch.zeros(source.size()).long()
			word_starts = []
			skip_cur_word = True
			for i in range(1, len(raw_to_word_align)):
				if raw_to_word_align[i-1] == raw_to_word_align[i]:
					# not a word start, as they align to the same word
					if not skip_cur_word and not _is_chinese_char(raw_tokens[i]):
						word_starts.pop(-1)
						skip_cur_word = True
					continue
				else:
					is_word_start[i] = 1
					if _is_chinese_char(raw_tokens[i]):
						word_starts.append(i)
						skip_cur_word = False
			is_word_start[0] = 0
			is_word_start[-1] = 0
			word_starts = torch.tensor(word_starts).long().view(-1, 1) # unused
			# 据我观察，这个word_starts记录的是所有中文的全词（不包括英文、数字或者标点），然后好像没啥用
		else:
			raw_tokens = self.tokenizer.convert_ids_to_tokens(source)
			is_word_start = torch.zeros(source.size()).long()
			for i, tok in enumerate(raw_tokens):
				if not (len(tok) > 2 and tok[:2] == "##"):
					is_word_start[i] == 1
		return is_word_start

	def add_whole_word_mask(self, source, mask_ratio, random_ratio, replace_length, chinese_ref=None):
		# 改编自fairseq(以及复旦NLP开源的CPT)中BART的加噪函数
		# 注意！BERT的词表从106开始才是正常词汇，所以这里也从106开始
		# 注意！输入的序列应当是没有PADDING的，并且携带开始和结束符
		# 注意！如果replace_length为0，这个函数实际上用于删除而非掩码

		assert replace_length in [-1, 0, 1]

		is_word_start = self.word_starts(source, chinese_ref=chinese_ref)

		num_to_mask = int(math.ceil(is_word_start.float().sum() * mask_ratio))
		if num_to_mask == 0:
			return source

		if self.mask_span_distribution is not None:
			lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

			# Make sure we have enough to mask
			cum_length = torch.cumsum(lengths, 0)
			while cum_length[-1] < num_to_mask:
				lengths = torch.cat(
					[
						lengths,
						self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
					],
					dim=0,
				)
				cum_length = torch.cumsum(lengths, 0)

			# Trim to masking budget
			i = 0
			while cum_length[i] < num_to_mask:
				i += 1
			lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
			num_to_mask = i + 1
			lengths = lengths[:num_to_mask]

			# 因为我所设置的泊松分布右移了一位，所以采样的所有长度都大于1
			# lengths = lengths[lengths > 0]
			# num_inserts = num_to_mask - lengths.size(0)
			# num_to_mask -= num_inserts

			assert (lengths > 0).all()
		else:
			lengths = torch.ones((num_to_mask,)).long()

		assert is_word_start[-1] == 0
		word_starts = is_word_start.nonzero(as_tuple=False)
		indices = word_starts[
			torch.randperm(word_starts.size(0))[:num_to_mask]
		].squeeze(1)
		# 在num_to_mask个token中，取mask_random个用随机词做替换，其余的用<MASK>做替换
		mask_random = torch.FloatTensor(num_to_mask).uniform_() < random_ratio
		source_length = source.size(0)
		assert source_length - 1 not in indices
		to_keep = torch.ones(source_length, dtype=torch.bool)
		is_word_start[
			-1
		] = 255  # acts as a long length, so spans don't go over the end of doc
		if replace_length == 0:
			to_keep[indices] = 0
		else:
			# keep index, but replace it with [MASK]
			# 在num_to_mask个token中，取mask_random个用随机词做替换，其余的用<MASK>做替换
			source[indices] = self.mask_index
			# source[indices[mask_random]] = torch.randint(
			# 	2072, 8843, size=(mask_random.sum(),)
			# ) # mBERT中的汉字区间
			source[indices[mask_random]] = torch.randint(
				self.LOWER_BOUND, self.UPPER_BOUND, size=(mask_random.sum(),)
			)

		if self.mask_span_distribution is not None:
			assert len(lengths.size()) == 1
			assert lengths.size() == indices.size()
			lengths -= 1
			while indices.size(0) > 0:
				assert lengths.size() == indices.size()
				lengths -= is_word_start[indices + 1].long()
				uncompleted = lengths >= 0
				indices = indices[uncompleted] + 1
				mask_random = mask_random[uncompleted]
				lengths = lengths[uncompleted]
				if replace_length != -1:
					# delete token
					to_keep[indices] = 0
				else:
					# keep index, but replace it with [MASK]
					source[indices] = self.mask_index
					# source[indices[mask_random]] = torch.randint(
					# 	2072, 8843, size=(mask_random.sum(),)
					# ) # mBERT中的汉字区间
					source[indices[mask_random]] = torch.randint(
						self.LOWER_BOUND, self.UPPER_BOUND, size=(mask_random.sum(),)
					)

		else:
			# 这个函数的意义在于，对于所有的non_word_start进行对应的MASK
			while indices.size(0) > 0:
				uncompleted = is_word_start[indices + 1] == 0
				indices = indices[uncompleted] + 1
				mask_random = mask_random[uncompleted]
				if replace_length != -1:
					# delete token
					# 如果replace_length不等于-1，那就对所有的non_word_start进行了删除
					to_keep[indices] = 0
				else:
					# keep index, but replace it with [MASK]
					source[indices] = self.mask_index
					# source[indices[mask_random]] = torch.randint(
					# 	2072, 8843, size=(mask_random.sum(),)
					# )
					source[indices[mask_random]] = torch.randint(
						self.LOWER_BOUND, self.UPPER_BOUND, size=(mask_random.sum(),)
					)

				assert source_length - 1 not in indices

		source = source[to_keep]

		return source

	def add_permuted_noise(self, tokens, permute_ratio):
		# 改编自fairseq中BART的加噪函数
		num_words = len(tokens)
		num_to_permute = math.ceil(((num_words * 2) * permute_ratio) / 2.0)
		substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
		tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
		return tokens

	def add_insertion_noise(self, tokens, insert_ratio, random_ratio):
		# 改编自fairseq中BART的加噪函数
		# 注意！BERT的词表从106开始才是正常词汇，所以这里也从106开始
		# 注意！输入序列应当是没有PADDING的，并且携带开始和结束符

		if insert_ratio == 0.0:
			return tokens

		num_tokens = len(tokens)
		n = int(math.ceil(num_tokens * insert_ratio))

		noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
		noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
		noise_mask[noise_indices] = 1
		result = torch.LongTensor(n + len(tokens)).fill_(-1)

		num_random = int(math.ceil(n * random_ratio))
		result[noise_indices[num_random:]] = self.mask_index
		# result[noise_indices[:num_random]] = torch.randint(
		# 	low=2072, high=8843, size=(num_random,)
		# ) # mBERT中的汉字区间
		result[noise_indices[:num_random]] = torch.randint(
			low=self.LOWER_BOUND, high=self.UPPER_BOUND, size=(num_random,)
		)

		result[~noise_mask] = tokens

		assert (result >= 0).all()
		return result

	def __call__(self, examples, args):

		if isinstance(examples[0], (dict, BatchEncoding)):
			input_ids = [e["input_ids"] for e in examples]
		else:
			input_ids = examples
			examples = [{"input_ids": e} for e in examples]

		features = []
		for e in examples:

			if 'src' not in e.keys():
				target = torch.LongTensor(e["input_ids"])

			else:
				sep_pos = e['input_ids'].index(self.tokenizer.sep_token_id)
				source = e['input_ids'][:sep_pos]
				target = e['input_ids'][sep_pos:]
				source = torch.LongTensor(source)
				target = torch.LongTensor(target)

			if 'ratio' in e.keys():
				mask_ratio = args.mask_ratio * e['ratio']
				if mask_ratio>=1 :
					print("WARNING: Mask ratio bigger than 1! Reset to 0.99!")
					mask_ratio = 0.99

				insert_ratio = args.inde_ratio * e['ratio']
				if insert_ratio>=1 :
					raise Exception("Insert ratio bigger than 1!")

				delete_ratio = args.inde_ratio * e['ratio']
				if delete_ratio>=1 :
					raise Exception("Delete ratio bigger than 1!")
			else:
				mask_ratio = args.mask_ratio
				insert_ratio = args.inde_ratio
				delete_ratio = args.inde_ratio			

			mask_random_ratio = args.random_ratio
			insert_random_ratio = args.random_ratio

			if mask_ratio > 0:
				if args.mask_strategy == "mask_to_multi":
					replace_length = -1
				else:
					replace_length = 1

				if args.model_type == "cBERT-HIT":
					# 如果你使用的BERT是哈工大-讯飞版的，就需要用ltp分词器切分
					target = self.add_whole_word_mask(target, 
													  mask_ratio=mask_ratio,
													  random_ratio=mask_random_ratio,
													  replace_length=replace_length,
													  chinese_ref=e["chinese_ref"])
				else:
					target = self.add_whole_word_mask(target, 
													  mask_ratio=mask_ratio,
													  random_ratio=mask_random_ratio,
													  replace_length=replace_length)

			if delete_ratio > 0:
				target = self.add_whole_word_mask(target,
												  mask_ratio=delete_ratio,
												  random_ratio=-1, #unused
												  replace_length=0)

			if insert_ratio > 0:
				target = self.add_insertion_noise(target, 
												  insert_ratio=insert_ratio,
												  random_ratio=insert_random_ratio)

			if 'src' not in e.keys():
				input_ids = target
				features.append({"input_ids": input_ids})
			else:
				input_ids = torch.cat([source, target])

				if args.model_type in ["cBERT-HIT", "cBERT", "BERT"]:
					source_token_type_ids = source.fill_(0)
					target_token_type_ids = target.fill_(1)
					target_token_type_ids[0] = 0 # SEP符号理应算前半句话
					token_type_ids = torch.cat([source_token_type_ids, target_token_type_ids])

					features.append({"input_ids": input_ids, "token_type_ids": token_type_ids})

				else:
					features.append({"input_ids": input_ids})

		features = self.tokenizer.pad(features)

		return features