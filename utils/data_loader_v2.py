'''
## Data Loader ##
# Creates a data generator which loads and preprocesses the demonstrations into pairs of ranked trajectories
@author: Mark Sinton (msinto93@gmail.com) 
'''
import IPython
import tensorflow as tf
import numpy as np
import os

class DataGenerator:
    def __init__(self, ob_shape, data, batch_size, traj_len, n_workers, preprocessing_offline):
        self.traj_len = traj_len
        self.ob_shape = ob_shape
                    
        # self.np_list = self.list_np_files(data_dir)

        self.minlen = data[0].shape[0]
                
        data = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
        
        # data = data.shuffle(self.minlen).repeat()
        data = data.shuffle(self.minlen).repeat()

        # The demonstrations data has already been preprocessed into trajectory pairs
        # data = data.batch(traj_len, drop_remainder=True)  # if tf version >= 1.10
        # data = data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size)) # if tf version < 1.10
            
        # Run preprocess function on trajectory pairs (filtering out invalid examples where reward values are equal), and prefetch parsed samples into buffer (must use tf.py_func as we need a python function to load the np files)
        data = data.map(lambda np_sample1, np_sample2: tf.py_func(self._process_trajectory_pairs,
                                                                  [np_sample1, np_sample2], [tf.float32, tf.float32]),
                        num_parallel_calls=n_workers).prefetch(buffer_size=5*batch_size)

        # Batch samples together
        data = data.batch(batch_size, drop_remainder=True) # if tf version >= 1.10
        # data = data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size)) # if tf version < 1.10
  
        self.data = data
        
    def __len__(self):
        return self.minlen
    
    def _process_trajectory_pairs(self, pair1, pair2):
        """Preprocessing function for already-generated trajectory pairs"""
        
        self.low_reward_snippet = pair1.copy()
        self.high_reward_snippet = pair2.copy()
        
        # Convert int to float and normalise to [0.0, 1.0] range
        self.low_reward_snippet = self.low_reward_snippet.astype(np.float32, copy=False)
        # low_reward_snippet /= 255.0
        self.high_reward_snippet = self.high_reward_snippet.astype(np.float32, copy=False)
        # high_reward_snippet /= 255.0
        
        return self.low_reward_snippet, self.high_reward_snippet
        
               
    def _process_demonstrations(self, demonstrations_files):
        """Preprocessing function for demonstrations - Generates preprocessed trajectory pairs from demonstrations"""
               
        filenames = [demonstrations_files[0].decode('UTF-8'), demonstrations_files[1].decode('UTF-8')]
    
        # Sort filenames into [lower_reward, higher_reward] order
        filenames = sorted(filenames, key=lambda x: self.extract_reward(x))
            
        low_reward = self.extract_reward(filenames[0])
        high_reward = self.extract_reward(filenames[1])

        # print("LLLLLLLLLLLLLLow name: {} high name: {}".format(low_reward, high_reward))
        # print("LLLLLLLLLLLLLLow shape: {} high shape: {}".format(low_reward.shape, high_reward.shape))

        # Only process samples further if they do not have equal rewards (as samples with equal rewards will be discarded later)
        if low_reward != high_reward:
            low_reward_traj = np.load(filenames[0])['states'].copy()
            high_reward_traj = np.load(filenames[1])['states'].copy()
            # print("LLLLLLLLLLLLLLow shape: {} high shape: {}".format(low_reward_traj.shape, high_reward_traj.shape))
            
            # Randomly sample start indices of snippets (start_index_high must be > start_index_low)
            start_index_high = np.random.randint(4, high_reward_traj.shape[0]-self.traj_len)
            start_index_low = np.random.randint(3, min(start_index_high, low_reward_traj.shape[0]-self.traj_len))
            
            # Extract snippets, stacking previous 3 frames alongside current frame
            # low_reward_snippet = np.concatenate((np.expand_dims(low_reward_traj[start_index_low-3:start_index_low+self.traj_len-3], axis=3),
            #                                      np.expand_dims(low_reward_traj[start_index_low-2:start_index_low+self.traj_len-2], axis=3),
            #                                      np.expand_dims(low_reward_traj[start_index_low-1:start_index_low+self.traj_len-1], axis=3),
            #                                      np.expand_dims(low_reward_traj[start_index_low:start_index_low+self.traj_len], axis=3)), axis=3)
            #
            # high_reward_snippet = np.concatenate((np.expand_dims(high_reward_traj[start_index_high-3:start_index_high+self.traj_len-3], axis=3),
            #                                      np.expand_dims(high_reward_traj[start_index_high-2:start_index_high+self.traj_len-2], axis=3),
            #                                      np.expand_dims(high_reward_traj[start_index_high-1:start_index_high+self.traj_len-1], axis=3),
            #                                      np.expand_dims(high_reward_traj[start_index_high:start_index_high+self.traj_len], axis=3)), axis=3)
            
            # Set top 10 rows to 0 to mask game score
            # low_reward_snippet[:, 0:10, :, :] = 0
            # high_reward_snippet[:, 0:10, :, :] = 0
    
            # Convert int to float and normalise to [0.0, 1.0] range
            low_reward_snippet = low_reward_traj[start_index_low:start_index_low+self.traj_len]\
                .astype(np.float32, copy=False)
            # low_reward_snippet = low_reward_snippet.astype(np.float32, copy=False)
            # low_reward_snippet /= 255.0
            high_reward_snippet = high_reward_traj[start_index_high:start_index_high+self.traj_len]\
                .astype(np.float32, copy=False)
            # high_reward_snippet = high_reward_snippet.astype(np.float32, copy=False)
            # high_reward_snippet /= 255.0
           
        else:
            # Create array of nans for the invalid samples (where rewards are equal), these will be filtered out afterwards            
            # low_reward_snippet = np.full((self.traj_len, 84, 84, 4), np.nan, dtype=np.float32)
            # high_reward_snippet = np.full((self.traj_len, 84, 84, 4), np.nan, dtype=np.float32)
            low_reward_snippet = np.full((self.traj_len, self.ob_shape), np.nan, dtype=np.float32)
            high_reward_snippet = np.full((self.traj_len, self.ob_shape), np.nan, dtype=np.float32)

        return low_reward_snippet, high_reward_snippet, low_reward, high_reward

    @staticmethod
    def list_np_files(data_dir):
        items = os.listdir(data_dir)
        data_list = []
        for item in items:
            data_list.append(data_dir + '/' + item)
        return data_list
    
    @staticmethod
    def extract_reward(filename):
        """Extract reward value from filename"""
        reward = filename.split('_')[-1].split('.')[:-1]
        return float('.'.join(reward))      
    
    
if __name__ == '__main__':
    ### For testing ###
    import cv2
    
    batch_sz = 1
    datagen = DataGenerator(data_dir='../samples/Breakout/train_data', batch_size=batch_sz, traj_len=50, n_workers=8, preprocessing_offline=False)
            
    iterator = tf.data.Iterator.from_structure(datagen.data.output_types)
    low_reward_traj, high_reward_traj, low_reward_value, high_reward_value = iterator.get_next()
    
    init_op = iterator.make_initializer(datagen.data)
       
    sess = tf.Session()
    sess.run(init_op)
        
    for step in range(100):
        print(step)
        try:
            low_reward_batch, high_reward_batch = sess.run([low_reward_traj, high_reward_traj])
            
            print(low_reward_batch.shape)
            print(high_reward_batch.shape)
            assert low_reward_batch.shape == high_reward_batch.shape
                         
            for sample in range(low_reward_batch.shape[0]):
                for state in range(low_reward_batch.shape[1]):
                    cv2.imshow('Low_reward', low_reward_batch[sample, state, :, :, -1])
                    cv2.imshow('High_reward', high_reward_batch[sample, state, :, :, -1])
                    cv2.waitKey(100)

        except tf.errors.OutOfRangeError:
            # Need to catch out of range errors as we will have a varying number of training steps at each epoch depending on the number of invalid samples
            break
        


    
        
        
    
    