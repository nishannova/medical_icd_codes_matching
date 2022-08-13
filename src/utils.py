#Utils

import os
import logging as logger
import pytesseract
import bs4
import re
import pandas as pd
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import  sent_tokenize

def rename_images(image_path, folder_path):
    try:
        page_num = int(re.findall(r"(\d+)\.[a-z]{3}$", image_path)[0])
        new_path = os.path.join(folder_path, f"page{page_num}.png")
        os.rename(image_path, new_path)
        doc_name = folder_path.rstrip("/").split("/")[-1]
        logger.info(f"Saved Image Number {page_num} of {doc_name} at {new_path}")
    except Exception as e:
        logger.error(f"Error in renaming image: {e}")
        
def ocr_hocr(img_path, psm=4):
    return pytesseract.image_to_pdf_or_hocr(
        img_path, extension="hocr", config=f"--psm {psm}"
    )

def get_text_from_hocr(hocr):
    hocr_text = []
    soup = bs4.BeautifulSoup(hocr, "html.parser")
    for line in soup.select(".ocr_line"):
        line_text = re.sub(r"\s+", " ", line.text).strip()
        alphanum_text = re.sub("[^a-zA-Z0-9 \n]", "", line_text).strip()
        if alphanum_text != "":
            hocr_text.append(line_text)
    return "\n".join(hocr_text)

def get_ocr_df(page_num, hocr):
    page_dict = {
        "page": '$page_num',
        "line": '$line',
        "text": '$text',
    }
    line = 1
    text = get_text_from_hocr(hocr).split(".")
    text = [re.sub(r"\n\d+", "", txt).replace("\n", " ") for txt in text]
    text = list(filter(None, text))
    
    page_record = list()
    
    for txt in text:
        if txt.isdigit():
            continue
        line += 1
        page_dict["page"] = page_num
        page_dict["line"] = line
        page_dict["text"] = txt
        page_record.append(page_dict.copy())
    return pd.DataFrame.from_records(page_record)

def ClinicalBert_embeddings(text, tokenizer, model):
    # tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    # model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    outputs = model(input_ids)
    embeddings_of_last_layer = outputs[0]
    cls_embeddings = embeddings_of_last_layer[0][0]
    
    return cls_embeddings

# stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    tag = None
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    # text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)     # /w* Matches Unicode word characters; 
    text = re.compile('[/(){}\[\]\|@,;]').sub(' ', text)
    text = re.compile('[^0-9a-z #+_]').sub('', text)  
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    tokenized = sent_tokenize(text)
    for i in tokenized:
        words = nltk.word_tokenize(i)
        tag = nltk.pos_tag(words)
    l = ['NNP', 'NNS', 'NN', 'RB', 'JJ', 'VBG', "CD"]
    text_new = []
    if tag:
        for i in tag:
            if i[1]  in l:
                text_new.append(i[0])

    text = [lemmatizer.lemmatize(word) for word in text_new]    ## POS tagger ###Change to lemmetization 
    text=" ".join(text)
    return text

