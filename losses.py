import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.KLLoss = nn.KLDivLoss()

	def forward(self, output, target):
		'''
		Output: (N,*) \n
		Target: (N,*) \n
		'''
		output = torch.log(output)  # Invert softmax
		# target = torch.log(target) # Invert softmax
		# How output distribution differs from target distribution
		return self.KLLoss(output, target)


class CELoss(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index)

	def forward(self, output, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		# output = torch.log(output) # torch.log(torch.clamp(output, min=1e-6))  # Invert softmax
		output = output.reshape(-1, output.shape[-1])  # (*,C)
		target = target.reshape(-1).long()  # (*)
		return self.CELoss(output, target)


class BCELoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.BCELoss = nn.BCEWithLogitsLoss()

	def forward(self, output, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		output = output.reshape(-1)  # (*,C)
		target = target.reshape(-1)  # (*)
		return self.BCELoss(output, target)


class RegLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.RegLoss = nn.MSELoss(reduction='mean')

	def forward(self, output, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		output = output[:, :, 1]
		target = target[:, :, 1]
		return self.RegLoss(output, target)


class CELossSame(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index)

	def forward(self, outputs, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		output_img = torch.log(outputs[0]) # Invert softmax
		output_txt = torch.log(outputs[1])
		output_sen = torch.log(outputs[2])

		output_img = output_img.reshape(-1, output_img.shape[-1]) # (*,C)
		output_txt = output_txt.reshape(-1, output_txt.shape[-1]) # (*,C)
		output_sen = output_sen.reshape(-1, output_sen.shape[-1]) # (*,C)
		target = target.reshape(-1).long() # (*)
		return self.CELoss(output_img, target) + self.CELoss(output_txt, target) + self.CELoss(output_sen, target)


class CELossShift(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss(ignore_index=ignore_index)

	def forward(self, output, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		output = output[:, :-1, :]# (* - 1,C)
		target = target[:,1:] # (* - 1)
		return self.CELoss(output, target)


class CELossTotal(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss(ignore_index=ignore_index)
		self.CELossShift = CELossShift(ignore_index=ignore_index)

	def forward(self, output, target):
		
		# print('gen:',self.CELossShift(output[0], target[0]))
		# print('cls:',self.CELoss(output[1], target[1]))
		return self.CELossShift(output[0], target[0]), self.CELoss(output[1], target[1])


'''class CELossTotal(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.BCELoss = BCELoss()
		self.CELossShift = CELossShift(ignore_index=ignore_index)

	def forward(self, output, target):
		print('gen:',self.CELossShift(output[0], target[0]))
		print('cls:',self.CELoss(output[1], target[1]))
		return self.CELossShift(output[0], target[0]) + self.BCELoss(output[1], target[1])'''


'''class CELossTotal(nn.Module):
	def __init__(self, ignore_index=-1, lambda_reg=0.001):
		super().__init__()
		self.CELoss = CELoss()
		self.RegLoss = RegLoss()
		self.CELossShift = CELossShift(ignore_index=ignore_index)
		self.lambda_reg = lambda_reg

	def forward(self, output, target):
		return self.CELossShift(output[0], target[0]) + self.CELoss(output[1], target[1]) + \
			   self.lambda_reg * self.RegLoss(output[2], output[1])'''


class CCELossTotal(nn.Module):
	def __init__(self, ignore_index=-1, cap_cls_weight=2, contrastive_weigt=1):
		super().__init__()
		self.CELoss = CELoss()
		self.CELossShift = CELossShift(ignore_index=ignore_index)
		self.ce = F.cross_entropy
		self.cap_cls_weight = cap_cls_weight
		self.contrastive_weight = contrastive_weigt

	def forward(self, output, target):
		contrastive_labels = torch.arange(output[2].shape[0], device=output[0].device)
		contrastive_loss = (self.ce(output[2], contrastive_labels) + self.ce(output[2].t(), contrastive_labels)) * 0.5
		total_loss = self.cap_cls_weight * (self.CELossShift(output[0], target[0]) + self.CELoss(output[1], target[1])) + \
					 (self.contrastive_weight * contrastive_loss)
		return total_loss


class CELossTotalEval(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss()
		self.CELossShift = CELossShift(ignore_index=ignore_index)

	def forward(self, output, target):
		return self.CELossShift(output[0], target[0]) + self.CELoss(output[1], target[1]) + self.CELoss(output[2], target[1])


class CELossTransfer(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss()
		self.CELossShift = CELossShift(ignore_index=ignore_index)

	def forward(self, output, target):
		return self.CELossShift(output[0], target[0]) # + self.CELoss(output[1], target[1])