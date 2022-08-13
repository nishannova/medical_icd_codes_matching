import os
import pandas as pd

from split_doc import split_doc_into_pages
from pre_process import get_processed_ocr_df
from embedding import persist_embedding
from match import match_similarity
import time

import logging as logger 
logger.getLogger().setLevel(logger.INFO)

from config import RESULT_FOLDER

def run():
    start = time.time()
    split_doc_into_pages()
    logger.info(f"\nSPLITTING DOC TOOK: {time.time() - start} Secs.\n")
    start_1 = time.time()
    ocr_df = get_processed_ocr_df()
    print(f"OCR HEAD\n{ocr_df.head()}")
    logger.info(f"\nBUILDING OCR DATA-FRAME TOOK: {time.time() - start_1} Secs.\n")
    start_1 = time.time()
    persist_embedding()
    logger.info(f"\nPERSISTING EMBEDDINGS OF PICKLE TOOK: {time.time() - start_1} Secs.\n")
    start_1 = time.time()
    logger.info("\nBEGINING SIMILARITY MATCHING\n")
    matched_res = match_similarity(ocr_df)
    logger.info(f"\nSIMILARITY MATCHING TOOK: {time.time() - start_1} Secs.\n")
    output_file = os.path.join(RESULT_FOLDER, "matched_res.csv")
    logger.info(f"\nENTIRE PROCESSING TOOK: {time.time() - start} Secs.\n")
    matched_res.to_csv(output_file)

if __name__ == "__main__":
    run()