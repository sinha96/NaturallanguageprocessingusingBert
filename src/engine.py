from tqdm import tqdm
import torch
import torch.nn as nn


def loss_func(outputs, targets):
	return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train(data_loader, model, optimizer, device, accumulation_step, scheduler):
	model.train()
	for batch_idx, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
		ids = batch_data['ids']
		token_type_ids = batch_data['token_type_ids']
		mask = batch_data['mask']
		targets = batch_data['targets']

		ids = ids.to(device, dtype=torch.long)
		token_type_ids = token_type_ids.to(device, dtype=torch.long)
		mask = mask.to(device, dtype=torch.long)
		targets = targets.to(device, dtype=torch.float)

		optimizer.zero_grad()
		output = model(
			ids=ids,
			mask=mask,
			token_type_ids=token_type_ids
		)

		loss = loss_func(outputs=output, targets=targets)
		loss.backward()

		if (batch_idx+1) % accumulation_step:
			optimizer.step()
			scheduler.step()


def eval_model(data_loader, model, device):
	model.eval()
	fin_targets = []
	fin_outputs = []
	with torch.no_grad():
		for batch_idx, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
			ids = batch_data['ids']
			token_type_ids = batch_data['token_type_idx']
			mask = batch_data['mask']
			targets = batch_data['targets']

			ids = ids.to(device, dtype=torch.long)
			token_type_ids = token_type_ids.to(device, dtype=torch.long)
			mask = mask.to(device, dtype=torch.long)
			targets = targets.to(device, dtype=torch.float)

			output = model(
				ids=ids,
				mask=mask,
				token_type_ids=token_type_ids
			)
			fin_targets.extend(targets.cpu().detach().numpy().to_list())
			output.extend(torch.sigmoid(output).cpu().detach().numpy().to_list())
	return fin_outputs, fin_targets
