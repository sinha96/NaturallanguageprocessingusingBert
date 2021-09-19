import torch
import config


class BERTDataset:
	def __init__(self, news, target):
		self.news = news
		self.target = target
		self.tokenizer = config.TOKENIZER
		self.max_len = config.MAX_LEN

	def __len__(self):
		return len(self.news)

	def __getitem__(self, item):
		news = str(self.news)
		news = ' '.join(news.split())
		inputs = self.tokenizer.encode_plus(
			news,
			None,
			add_special_tokens=True,
			max_length=self.max_len
		)
		ids = inputs['input_ids']
		mask = inputs['attention_mask']
		token_type_ids = inputs['token_type_ids']

		padding_len = self.max_len - len(ids)
		ids += ([0] * padding_len)
		mask += ([0] * padding_len)
		token_type_ids += ([0] * padding_len)

		return {
			'ids': torch.tensor(ids, dtype=torch.long),
			'mask': torch.tensor(mask, dtype=torch.long),
			'token_type_ids': torch.tensor(token_type_ids, dtype=torch.float),
			'targets': torch.tensor(self.target[item], dtype=torch.float)
		}
