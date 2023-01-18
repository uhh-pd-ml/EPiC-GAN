import numpy as np
from torch.utils.data import IterableDataset


# loads a dataset from a numpy file and passes it to buckets of 1 length with max batch size
# improved with shuffling batches 
# assumse data in form [batch, feats, points]
class Dataset_Bucketing(IterableDataset):
    
    def __init__(self, npy_file, batch_size_max, shuffle=True):
        self.whole_set = np.load(npy_file)
        self.batch_size_max = batch_size_max
        self.shuffle = shuffle
        
    def __len__(self):
        ls = self.calc_lengths(self.whole_set)
        _, uni_frqs = np.unique(ls, return_counts=True)
        divisors = uni_frqs // self.batch_size_max
        return divisors.sum() + np.count_nonzero(uni_frqs % self.batch_size_max)
    
    def calc_lengths(self, data):   # how many particles per jet
        lengths = np.count_nonzero(data[...,0], axis=1)
        return lengths
        
    def make_bucket_dict(self, data, batch_size=128):
        ls = self.calc_lengths(data)
        uni_ls = np.unique(ls)
        bucket_dict = {}
        bucket_ids = []
        for i in range(len(uni_ls)):
            bucket = data[ls == uni_ls[i]]
            bucket = bucket[:,0:uni_ls[i],:]  # drop all zero padded values, assuming [batch, points, feats]
            sub_bucket_dict, bucket_id_ary = self.make_sub_buckets(bucket, batch_size=batch_size, bucket_id=i)
            bucket_dict[i] = sub_bucket_dict
            bucket_ids.append(bucket_id_ary)  # tupel of bucket and sub_bucket index
        bucket_ids = np.vstack(bucket_ids)
        return bucket_dict, bucket_ids

    def make_sub_buckets(self, data, batch_size=128, bucket_id=0):
        n_samples = len(data)
        n_batches = n_samples // batch_size
        if n_samples % batch_size != 0: # for the case of excess events not filling the whole batch_size
            n_batches +=1
        sub_bucket_dict = {}
        i = 0
        bucket_id_list = []
        for j in range(n_batches):
            sub_bucket = data[i:i+batch_size]
            sub_bucket_dict[j] = sub_bucket
            i += batch_size
            bucket_id_list.append([bucket_id, j])
        bucket_id_ary = np.vstack(bucket_id_list)
        return sub_bucket_dict, bucket_id_ary
    
    def get_batch(self):
        # get unique lengths of dataset and shuffle them
        if self.shuffle:
            permutation_dataset = np.random.permutation(len(self.whole_set))
            data = self.whole_set[permutation_dataset]
        else:
            data = self.whole_set
        
        bucket_dict, bucket_ids = self.make_bucket_dict(data, batch_size=self.batch_size_max)
        if self.shuffle:
            permutation = np.random.permutation(len(bucket_ids))
            bucket_ids = bucket_ids[permutation]
        
        i = 0  # bucket/sub_bucket index
        while True:
            if i >= len(bucket_ids):    # resetting the generator --> shuffel dataset, shuffle lengths
                if self.shuffle:
                    permutation_dataset = np.random.permutation(len(data))
                    data = data[permutation_dataset]

                bucket_dict, bucket_ids = self.make_bucket_dict(data, batch_size=self.batch_size_max)
                if self.shuffle:
                    permutation = np.random.permutation(len(bucket_ids))
                    bucket_ids = bucket_ids[permutation]
                i = 0
            else:    # normally, not resetting mode
                while i < len(bucket_ids):
                    j,k = bucket_ids[i]
                    batch = bucket_dict[j][k]
                    i += 1
                    yield batch

    def __iter__(self):  
        return self.get_batch()
