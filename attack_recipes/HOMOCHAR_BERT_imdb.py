#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime


# In[2]:


t1=datetime.now()


# In[3]:


get_ipython().system('pip install textattack')


# In[4]:


get_ipython().system('pip install tensorflow-text')


# In[5]:


import textattack
import transformers


# In[6]:


model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)


# In[7]:


get_ipython().system('pip install torchfile')


# In[8]:


# Construct our four components for `Attack`

from textattack.constraints.pre_transformation import RepeatModification, StopwordModification

from textattack.constraints.semantics import WordEmbeddingDistance


# In[9]:


goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)

constraints = [

    RepeatModification(),
    StopwordModification(),
    WordEmbeddingDistance(min_cos_sim=0.7)

]


# In[10]:


transformation = textattack.transformations.word_swaps.word_swap_homoglyph_swap.WordSwapHomoglyphSwap()

search_method = textattack.search_methods.beam_search.BeamSearch(beam_width=8)


# In[11]:


attack =  textattack.Attack(goal_function, constraints, transformation, search_method)

dataset = textattack.datasets.HuggingFaceDataset("imdb", dataset_columns=(["text"],'label'))


# In[ ]:


print(attack)


# In[12]:


attack_args = textattack.AttackArgs(

    num_examples=10,

    log_to_csv="HOMOCHAR_BERT_imdb.csv",

    checkpoint_interval=2,

    checkpoint_dir="checkpoints",

    disable_stdout=True

)

attacker = textattack.Attacker(attack, dataset, attack_args)

attacker.attack_dataset() 


# In[ ]:


from textattack import Attacker

attacker = Attacker(attack, dataset)
attack_results=attacker.attack_dataset()


# In[13]:


t2=datetime.now()


# In[14]:


a=(t2-t1).total_seconds()


# In[15]:


print(a)

