from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import pandas as pd
import logging  as logger
logger.getLogger().setLevel(logger.INFO)

from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel

from config import PICKLE_FOLDER, CPT_CODE_FILE, BERT_PRETRAINED
from utils import ClinicalBert_embeddings,clean_text




def match_similarity(ocr_df):
    # new_df=ocr_df[~ocr_df.page.isin([1,10])]
    new_df=ocr_df.copy()
    cpt_df=pd.read_excel(CPT_CODE_FILE)
   
    new_df1 = new_df[~(new_df.text.str.contains("Non-VIP") | new_df.text.str.contains("Pharmacy Exclusions"))]
    tokenizer = AutoTokenizer.from_pretrained(BERT_PRETRAINED)
    model = AutoModel.from_pretrained(BERT_PRETRAINED)
    
    overall_code=[]
    
    
    
    for file_name in os.listdir(PICKLE_FOLDER):
        logger.warning(f"PROCESSING PICKLE: {file_name}")
        infile=open(os.path.join(PICKLE_FOLDER,file_name),"rb")
        pkl_dict=pickle.load(infile)
        infile.close()
        code=[]
        for _,row in new_df1.iterrows():
            sim_dict=dict()
            cls_emb=ClinicalBert_embeddings(row["text"], tokenizer, model) #TODO: Pre-process text
            for i in pkl_dict:
                s=cosine_similarity(cls_emb.detach().numpy().reshape(1,-1),pkl_dict[i].detach().numpy().reshape(1,-1))[0][0]
                sim_dict[i]=s
            code.append(list(sorted(sim_dict.items(),key=lambda x:x[1]))[-1])
        overall_code.append(code)

    final_code=[]
    final_text = []
    for num_text in range(len(overall_code[0])):
        L=[]
        for num_pkl in range(len(overall_code)):
            L.append(overall_code[num_pkl][num_text])
        final_code.append(sorted(L,key=lambda x:x[1])[-1][0])
        code = sorted(L,key=lambda x:x[1])[-1][0]
        final_text.append(str(cpt_df[cpt_df["CPT_CODE"]==code].FULL_DESCRIPTION.values[0]))

    new_df1["CODE"]=final_code
    new_df1["Matched_Description"]=final_text
    logger.info(f" SAMPLE MATCHING DATA FRAME: {new_df1.head()}")

    return new_df1