import torch
import torch.nn as nn
import numpy as np
from networks import get_network, set_requires_grad
from util.hausdorff import directed_hausdorff,local_directed_hausdorff
from agent.base import IMLEzEAgent
from sampler import gen_samples
#from util.emd import earth_mover_distance
from dciknn_cuda.dciknn_cuda import DCI
from torch.distributions import normal

class MainAgentImle(IMLEzEAgent):
	def __init__(self, config):
		super(MainAgentImle, self).__init__(config)

		self.weight_z_L1 = config.weight_z_L1
		self.weight_partial_rec = config.weight_partial_rec
		self.dci_db = DCI(128,2,10,100,10)
		self.mm = normal.Normal(0,4)

	def build_net(self, config):
		# load pretrained pointAE
		self.pointAE = get_network(config, "pointAE")
		try:
			ae_weights = torch.load(config.pretrain_ae_path)['model_state_dict']
		except Exception as e:
			raise ValueError("Check the path for pretrained model of point AE. \n{}".format(e))
		self.pointAE.load_state_dict(ae_weights)
		self.pointAE = self.pointAE.eval().cuda()
		set_requires_grad(self.pointAE, False)


		# build G, D
		self.netG = get_network(config, "G").cuda()
		self.l2_loss = nn.MSELoss()
		

	def collect_loss(self):
		# loss_dict = {"D_GAN": self.loss_D,
		#              "G_GAN": self.loss_G_GAN,
		#              "z_L1": self.loss_z_L1,
					 
		loss_dict = {
					"imle": self.l2,            
					 "partial_rec": self.loss_partial_rec}
		return loss_dict

	def forward(self, data,test=False):
		self.raw_pc = data['raw'].cuda()
		self.real_pc = data['real'].cuda()

		with torch.no_grad():
			self.raw_latent = self.pointAE.encode(self.raw_pc)
			self.real_latent = self.pointAE.encode(self.real_pc)
		if test != True:
			self.forward_GE()
		else:
			self.forward_GE_test()
			# print('test')
		
	def forward_GE_test(self):
		self.fake_latent_list = []
		
		
		# self.z_random = self.get_random_noise(self.raw_latent.size(0))
		self.z_random = self.mm.sample([self.raw_latent.size(0), self.z_dim*4]).cuda()
		self.fake_latent = self.netG(self.raw_latent, self.z_random)
		
		self.fake_pc = self.pointAE.decode(self.fake_latent)
		#self.z_rec, z_mu, z_logvar = self.netE(self.fake_pc)

	def forward_GE(self):
		self.fake_latent_list = []
		
		for idx in range(80):
			# self.z_random = self.get_random_noise(self.raw_latent.size(0))
			self.z_random = self.mm.sample([self.raw_latent.size(0), self.z_dim*4]).cuda()
			self.fake_latent = self.netG(self.raw_latent, self.z_random)
			# print(self.fake_latent.shape)
			self.fake_latent_list.append(self.fake_latent)

		self.fake_latent_list = torch.stack(self.fake_latent_list,1)
		
		
		#sample     
		self.fake_latent = gen_samples(self.dci_db,self.fake_latent_list,self.real_latent)
		self.fake_pc = self.pointAE.decode(self.fake_latent)
		#self.z_rec, z_mu, z_logvar = self.netE(self.fake_pc)

	def backward_D(self):
		# fake
		pred_fake = self.netD(self.fake_latent.detach())
		fake = torch.zeros_like(pred_fake).fill_(0.0).cuda()
		self.loss_D_fake = self.criterionGAN(pred_fake, fake)

		# real
		pred_real = self.netD(self.real_latent.detach())
		real = torch.ones_like(pred_real).fill_(1.0).cuda()
		self.loss_D_real = self.criterionGAN(pred_real, real)

		self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
		self.loss_D.backward()


	def backward_EG(self,data):
		
		f1 = self.pointAE.encode(self.real_pc)
		f2 = self.pointAE.encode(self.fake_pc)
	
		# 2. partial scan reconstruction
		
		self.l2 = self.l2_loss(self.fake_latent,self.real_latent)*5
		self.loss_partial_rec = local_directed_hausdorff(self.raw_pc, self.fake_pc) * self.weight_partial_rec * 1
		
		self.loss_EG =  self.loss_partial_rec + self.l2 

	
		self.loss_EG.backward()

	def update_G_and_E(self,data):
		#set_requires_grad(self.netD, False)
		self.optimizer_G.zero_grad()
		self.backward_EG(data)
		self.optimizer_G.step()

	def optimize_network(self,data):
		self.update_G_and_E(data)
		#self.update_D()

	def get_point_cloud(self):
		"""get real/fake/raw point cloud of current batch"""
		real_pts = self.real_pc.transpose(1, 2).detach().cpu().numpy()
		fake_pts = self.fake_pc.transpose(1, 2).detach().cpu().numpy()
		raw_pts = self.raw_pc.transpose(1, 2).detach().cpu().numpy()
		return real_pts, fake_pts, raw_pts

	def visualize_batch(self, data, mode, **kwargs):
		tb = self.train_tb if mode == 'train' else self.val_tb

		num = 4

		real_pts = data['real'][:num].transpose(1, 2).detach().cpu().numpy()
		fake_pts = self.fake_pc[:num].transpose(1, 2).detach().cpu().numpy()
		raw_pts = self.raw_pc[:num].transpose(1, 2).detach().cpu().numpy()

		fake_pts = np.clip(fake_pts, -0.999, 0.999)

		tb.add_mesh("real", vertices=real_pts, global_step=self.clock.step)
		tb.add_mesh("fake", vertices=fake_pts, global_step=self.clock.step)
		tb.add_mesh("input", vertices=raw_pts, global_step=self.clock.step)
