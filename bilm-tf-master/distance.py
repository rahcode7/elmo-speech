import os
import tensorflow as tf
import numpy as np
import scipy.spatial.distance as ds
import pandas as pd 

vocab_file = os.path.join('../swb/', 'vocab.txt')
options_file = os.path.join('../models/', 'options.json')
weight_file = os.path.join('../models/weights/', 'swb_weights.hdf5')
 
# Create a Batcher to map text to character ids.

from bilm import Batcher, BidirectionalLanguageModel, weight_layers
 
 
batcher = Batcher(vocab_file, 50)
 
# Input placeholders to the biLM.
context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
 
# Build the biLM graph.
bilm = BidirectionalLanguageModel(options_file, weight_file)
 
# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_character_ids)
 
# Get an op to compute ELMo (weighted average of the internal biLM layers)
elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
 
# 5 words
raw_context = ['talking',
                'movie',
                'music',
                'world',
                'television']
 
tokenized_context = [sentence.split() for sentence in raw_context]
print(tokenized_context)   


with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())
 
    # Create batches of data.
    context_ids = batcher.batch_sentences(tokenized_context)
    print("Shape of context ids = ", context_ids.shape)
 
    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_context_input_ = sess.run(
        elmo_context_input['weighted_op'],
        feed_dict={context_character_ids: context_ids}
    )
 
print("Shape of generated embeddings = ",elmo_context_input_.shape)

print(elmo_context_input_.shape)

# Euclidean distance 
eucl_df = pd.DataFrame(columns=['Word1','Word2','EuclideanDistance'])
#elmo_context_input[0,0,:]
for i in range(5):
    #print(i)
    # Get embeddings for that word
    others = [0,1,2,3,4]
    del others[i]
    for j in others: 
        #print(j)
        dist =  np.linalg.norm(elmo_context_input_[i,0,:]-elmo_context_input_[j,0,:])

        row = {'Word1': raw_context[i], 'Word2': raw_context[j], 'EuclideanDistance' : np.round(dist,2)}
        eucl_df =eucl_df.append(row,ignore_index=True)
print(eucl_df)
eucl_df.to_csv("../results/euclidean.csv",index=None)

# Cosine distance
cosine_df = pd.DataFrame(columns=['Word1','Word2','CosineDistance'])
#elmo_context_input[0,0,:]
for i in range(5):
    #print(i)
    # Get embeddings for that word
    others = [0,1,2,3,4]
    del others[i]
    for j in others: 
        #print(j)
        dist =  ds.cosine(elmo_context_input_[i,0,:],elmo_context_input_[j,0,:])
        row = {'Word1': raw_context[i], 'Word2': raw_context[j], 'CosineDistance' : np.round(dist,2)}
        cosine_df =cosine_df.append(row,ignore_index=True)
print(cosine_df)
cosine_df.to_csv("../results/cosine.csv",index=None)

