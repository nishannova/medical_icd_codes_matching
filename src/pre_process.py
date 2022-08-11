#Getting OCR Results

import pandas as pd
import os
import re
import logging as logger 
logger.getLogger().setLevel(logger.INFO)

from config import OUTPUT_FOLDER, ATTRIBUTES
from utils import ocr_hocr, get_ocr_df,clean_text

def get_processed_ocr_df():
    ocr_df = pd.DataFrame(columns=ATTRIBUTES)
    logger.info("PROCESSING BEGINS")
    for page_path in os.listdir(OUTPUT_FOLDER):
        page_num = int(re.findall(r'\d+', page_path)[0])
        page_img = os.path.join(OUTPUT_FOLDER, page_path)
        hocr = ocr_hocr(page_img).decode(encoding="utf-8", errors="ignore")
        ocr_df = pd.concat([ocr_df, get_ocr_df(page_num, hocr)])
    logger.info(f"PROCESSED DF: {ocr_df.head()}")
    ocr_df["text"] = ocr_df["text"].apply(clean_text)
    return ocr_df

