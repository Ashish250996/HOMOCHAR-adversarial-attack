#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime


# In[ ]:


t1=datetime.now()


# In[ ]:


get_ipython().system('pip install textattack')
import textattack
import transformers


# In[2]:


# Load model, tokenizer, and model_wrapper
import textattack

model=  textattack.models.helpers.word_cnn_for_classification.WordCNNForClassification.from_pretrained("cnn-mr")
import json
  
# Opening JSON file
f = open('/home/biometriclab20/ashish/vocab.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)
tokenizer = textattack.models.tokenizers.glove_tokenizer.GloveTokenizer(word_id_map=data, pad_token_id=400000)

model_wrapper = textattack.models.wrappers.pytorch_model_wrapper.PyTorchModelWrapper(model, tokenizer)


# In[3]:


# Construct our four components for `Attack`

from textattack.constraints.pre_transformation import RepeatModification, StopwordModification

from textattack.constraints.semantics import WordEmbeddingDistance


# In[4]:


goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)

constraints = [

    RepeatModification(),

    StopwordModification(),

    WordEmbeddingDistance(min_cos_sim=0.7)

]


# In[5]:


transformation = textattack.transformations.word_swaps.word_swap_homoglyph_swap.WordSwapHomoglyphSwap()

search_method = textattack.search_methods.beam_search.BeamSearch(beam_width=8)


# In[6]:


# Construct the actual attack
dataset = textattack.datasets.HuggingFaceDataset("rotten_tomatoes", split="test")

attack =  textattack.Attack(goal_function, constraints, transformation, search_method)


# In[7]:


print(attack)


# In[8]:


# Attack 20 samples with CSV logging and checkpoint saved every 5 interval
attack_args = textattack.AttackArgs(

    num_examples=500,

    log_to_csv="HOMOCHAR_cnn_MR.csv",

    checkpoint_interval=100,

    checkpoint_dir="checkpoints",

    disable_stdout=True

)

attacker = textattack.Attacker(attack, dataset, attack_args)

attacker.attack_dataset() 


# In[9]:


from textattack import Attacker

attacker = Attacker(attack, dataset)
attack_results=attacker.attack_dataset()


# In[ ]:


t2=datetime.now()


# In[ ]:


a=(t2-t1).total_seconds()


# In[ ]:


print(a)

