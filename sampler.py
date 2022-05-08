import torch
from dciknn_cuda.dciknn_cuda import DCI





def gen_samples(dci_db,fake_data,real_data):
	
	
	indices = []
	fake_data_sampled = []

	n_samples = fake_data.shape[0]
	#with torch.no_grad():
	for sample in range(n_samples):
		# print(sample)
		fake_data_sample = fake_data[sample]
		real_data_sample = torch.unsqueeze(real_data[sample],0)
		# print(fake_data_sample.shape,real_data_sample.shape)
		# try:
		dci_db.add(fake_data_sample)
		index,_ = dci_db.query(real_data_sample,1,5000)
		#print(index)
		indices.append(index)
		dci_db.clear()
		
		
		fake_data_sampled.append(fake_data_sample[index[0][0].long()])
		# except:
		# 	fake_data_sampled.append(fake_data_sample[0])
		# 	print('Exception handled')


			
		
	#dci_db.free()
	fake_data = torch.stack(fake_data_sampled)
	# print(fake_data.shape)
	torch.cuda.empty_cache()
	

	return fake_data