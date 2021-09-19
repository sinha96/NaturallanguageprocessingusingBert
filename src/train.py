import torch.utils.data
import config
import pandas as pd
import numpy as np
from sklearn import metrics
import engine
import dataset
from sklearn.model_selection import train_test_split
from model import BERTBasedUncased
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():
	df_t = pd.read_csv(config.TRAIN_FILE_TRUE).fillna('none')
	df_f = pd.read_csv(config.TRAIN_FILE_FAKE).fillna('none')
	df_t['target'] = 1
	df_f['target'] = 0
	df = pd.concat([df_t, df_f])
	df = df[['text', 'target']]
	df_train, df_valid = train_test_split(
		df,
		test_size=0.1,
		random_state=45,
		stratify=df.target.values,
		shuffle=True
	)
	df_train = df_train.reset_index(drop=True)
	df_valid = df_valid.reset_index(drop=True)

	train_dataset = dataset.BERTDataset(
		news=df_train.text.values,
		target=df_train.target.values
	)
	train_dataloader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=config.TRAIN_BATCH_SIZE,
		num_workers=3
	)

	valid_dataset = dataset.BERTDataset(
		news=df_valid.text.values,
		target=df_valid.target.values
	)
	valid_dataloader = torch.utils.data.DataLoader(
		valid_dataset,
		batch_size=config.VALID_BATCH_SIZE,
		num_workers=4
	)
	device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
	model = BERTBasedUncased()
	model.to(device)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerBias.weight']
	optimiser_parameters = [
		{
			'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
			'weight_decay': 0.001
		},
		{
			'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
			'weight_decay': 0.0
		}
	]
	num_train_step = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
	optimizer = AdamW(optimiser_parameters, lr=3e-5)
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=0,
		num_training_steps=num_train_step
	)
	best_accuracy = 0.0
	for epoch in range(config.EPOCHS):
		engine.train(
			data_loader=train_dataloader,
			model=model,
			optimizer=optimizer,
			device=device,
			scheduler=scheduler,
			accumulation_step=config.ACCUMULATION
		)
		outputs, targets = engine.eval_model(
			data_loader=valid_dataloader,
			model=model,
			device=device,
		)
		outputs = np.array(outputs) > 0.5
		accuracy = metrics.accuracy_score(targets, outputs)
		print(f"Accuracy Score = {accuracy}")
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			torch.save(
				model.state_dict(),
				config.MODEL_PATH
			)


if __name__ == '__main__':
	run()
