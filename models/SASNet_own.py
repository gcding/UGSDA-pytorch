import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


def upsample_bilinear(x, size):
	return F.interpolate(x, size=size[2:], mode='bilinear', align_corners=False)

class SASNet_own(nn.Module):
	"""
	Implement DGNet model
	"""

	def __init__(self):
		super(SASNet_own, self).__init__()

		# Backbone Part
		# frontend
		model = list(models.vgg16(pretrained=True).features.children())
		self.feblock = nn.Sequential(*model[:4])       # 64  1
		self.feblock1 = nn.Sequential(*model[4:16])    # 256 2
		self.feblock2 = nn.Sequential(*model[16:23])   # 512 4 
		self.feblock3 = nn.Sequential(*model[23:30])   # 512 8

		# backend
		self.beblock3 = nn.Sequential(
			nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(1024, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
		)
		self.beblock2 = nn.Sequential(
			nn.Conv2d(1024, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
		)
		self.beblock1 = nn.Sequential(
			nn.Conv2d(512, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
		)

		self.output_layer = nn.Sequential(
			nn.Conv2d(64, 1, kernel_size=1, bias=False),
			nn.ReLU(inplace=True)
			# nn.Dropout2d(p=0.5)
		)
		self.deblock = nn.Sequential(
			nn.Conv2d(896, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
		)
		
	def forward(self, x, style_hallucination=False, out_prob=False, return_style_features=False, teacher_mode=False):
		
		x_size = x.size()

		x = self.feblock(x)
		
		x = self.feblock1(x)
		x1 = x
		x = self.feblock2(x)
		x2 = x
		x = self.feblock3(x)
		# decoder 
		x = self.beblock3(x)

		x3_ = x
		x = upsample_bilinear(x, x2.shape)
		x = torch.cat([x, x2], 1)

		x = self.beblock2(x)
		x2_ = x
		x = upsample_bilinear(x, x1.shape)
		x = torch.cat([x, x1], 1)

		x1_ = self.beblock1(x)

		x2_ = upsample_bilinear(x2_, x1.shape)
		x3_ = upsample_bilinear(x3_, x1.shape)

		x = torch.cat([x1_, x2_, x3_], 1)

		x = self.deblock(x)
		
		x = self.output_layer(x)
		main_out = upsample_bilinear(x, size=x_size)

		
		return main_out