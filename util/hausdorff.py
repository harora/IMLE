import torch
import numpy as np

def directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
	"""

	:param point_cloud1: (B, 3, N)
	:param point_cloud2: (B, 3, M)
	:return: directed hausdorff distance, A -> B
	"""
	n_pts1 = point_cloud1.shape[2]
	n_pts2 = point_cloud2.shape[2]

	pc1 = point_cloud1.unsqueeze(3)
	pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
	pc2 = point_cloud2.unsqueeze(2)
	pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

	l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

	shortest_dist, _ = torch.min(l2_dist, dim=2)

	hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

	if reduce_mean:
		hausdorff_dist = torch.mean(hausdorff_dist)

	return hausdorff_dist


def local_directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
	"""

	:param point_cloud1: (B, 3, N)
	:param point_cloud2: (B, 3, M)
	:return: directed hausdorff distance, A -> B
	"""
	n_pts1 = point_cloud1.shape[2]
	n_pts2 = point_cloud2.shape[2]
	# print(point_cloud1.max(),point_cloud2.max())
	# print(point_cloud1.min(),point_cloud2.min())

	B = point_cloud1.shape[0]
	HD = []
	for pc in range(B):
		PC1 = point_cloud1[pc]
		PC2 = point_cloud2[pc]

		hd = torch.tensor(0.).to(PC1.device)
		
		x_min = PC1[0].min()
		x_d = PC1[0].max() - PC1[0].min()

		y_min = PC1[1].min()
		y_d = PC1[1].max() - PC1[1].min()

		z_min = PC1[2].min()
		z_d = PC1[2].max() - PC1[2].min()

		
		for i in range(4):
			for j in range(4):
				for k in range(4):
					

					ll = torch.tensor([x_min+0.25*i*x_d,     y_min+0.25*(j)*y_d,   z_min+0.25*(k)*z_d]).to(PC1.device)
					ur = torch.tensor([x_min+0.25*(i+1)*x_d, y_min+0.25*(j+1)*y_d, z_min+0.25*(k+1)*z_d]).to(PC1.device) 
					# print(ll,ur)
					
					inidx = torch.logical_and(ll <= PC1.transpose(1,0), PC1.transpose(1,0) <= ur)
					inidx = torch.all(inidx,dim=1)
					
					pc1 = PC1.transpose(1,0)[inidx].transpose(1,0)

					inidx = torch.logical_and(ll <= PC2.transpose(1,0), PC2.transpose(1,0) <= ur)
					inidx = torch.all(inidx,dim=1)


					pc2 = PC2.transpose(1,0)[inidx].transpose(1,0)
					# print(pc1.shape,pc2.shape,'lllll')
					

					if pc1.shape[1]>0 and pc2.shape[1]>0:
						# print(pc1.shape,pc2.shape,'f')
						pc1 = pc1.unsqueeze(0).unsqueeze(3)
						pc1 = pc1.repeat((1, 1, 1, pc2.shape[1])) # (B, 3, N, M)
						pc2 = pc2.unsqueeze(0).unsqueeze(2)
						pc2 = pc2.repeat((1, 1, pc1.shape[2], 1)) # (B, 3, N, M)

						l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

						shortest_dist, _ = torch.min(l2_dist, dim=2)

						hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )
						# print(hausdorff_dist)
						hd += hausdorff_dist[0]
						# print(hd,'hd')
		
		HD.append(hd)	
	# print(HD,'HD')
	if reduce_mean:
		HD = torch.mean(torch.tensor(HD))

	return HD
