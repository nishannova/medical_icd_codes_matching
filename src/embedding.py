#Dependencies
from pickle import HIGHEST_PROTOCOL
import pickle
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import re
import logging as logger
logger.getLogger().setLevel(logger.INFO)
import string
import os
import gc
import time
from utils import clean_text
from config import CPT_CODE_FILE, PICKLE_FOLDER, BERT_PRETRAINED


def persist_embedding():
    def ClinicalBert_embeddings(text):
        
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        outputs = model(input_ids)
        embeddings_of_last_layer = outputs[0]
        cls_embeddings = embeddings_of_last_layer[0][0]
        
        return cls_embeddings
    
    start = time.time()
    start_1 = time.time()
    cpt_df=pd.read_excel(CPT_CODE_FILE)
    
    ###### Preprocessing the Full Description column
    cpt_df["FULL_DESCRIPTION"] = cpt_df["FULL_DESCRIPTION"].apply(clean_text)
    
    
    
    logger.info(cpt_df.head())

    pickle_path = PICKLE_FOLDER
    tokenizer = AutoTokenizer.from_pretrained(BERT_PRETRAINED)
    model = AutoModel.from_pretrained(BERT_PRETRAINED)

    
    
    input_ids = torch.tensor(tokenizer.encode(cpt_df["FULL_DESCRIPTION"].values[0])).unsqueeze(0)
    outputs = model(input_ids)
    embeddings_of_last_layer=outputs[0]
    cls_embeddings=embeddings_of_last_layer[0][0]

    cpt_embeddings= dict()
    count = 1
    batch = 1
    start = time.time()
    for idx,row in cpt_df.iterrows():
        try:
            
            embeddings = ClinicalBert_embeddings(row["FULL_DESCRIPTION"]) #TODO: Pre-processed text inputs
        except Exception as ex:
            logger.error(f"Error: {ex} Occurred")
            continue
        cpt_embeddings[row["CPT_CODE"]] = embeddings
        if count % 500 == 0 and count > 1:        
            with open(os.path.join(pickle_path, str(batch)+".pickle"), "wb") as handle:
                pickle.dump(cpt_embeddings, handle)
            logger.warning(f"Dumped Batch: {batch} in: {time.time() - start_1} Secs.")
            start_1 = time.time()
            batch += 1
            del cpt_embeddings
            cpt_embeddings= dict()
            gc.collect()
        del embeddings
        count += 1
        if count % 500 == 0:
            logger.warning(F"PROCESSED:{count} records")
    with open(os.path.join(pickle_path, str(batch)+".pickle"), "wb") as handle:
        pickle.dump(cpt_embeddings, handle)
        logger.warning(f"Dumped Batch: {batch}")
        logger.warning(f"Processed : {count} records")
    logger.warning(f"\n\nPROCESSING TOOK: {time.time() - start} Secs")