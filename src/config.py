import os

#Configs

DATA_ROOT = "/Users/nishanali/WorkSpace/text_mapping/data"
RAW_FOLDER = os.path.join(DATA_ROOT, "raw")
doc_name = "Exclusions - non-VIP 5.2.pdf"
FILENAME = os.path.join(RAW_FOLDER, doc_name)
PROCESSED_FOLDER = os.path.join(DATA_ROOT, "processed")
OUTPUT_FOLDER = os.path.join(PROCESSED_FOLDER, doc_name.split(".")[0])
HELPER_FOLDER = os.path.join(DATA_ROOT, "helper")
CPT_CODE_FILE = os.path.join(HELPER_FOLDER, "CPT CODES.xlsx")
PICKLE_FOLDER = os.path.join(PROCESSED_FOLDER, "pickle")
RESULT_FOLDER = os.path.join(DATA_ROOT, "result")

ATTRIBUTES = ["page","line","text", "CODE"]

BERT_PRETRAINED = "bvanaken/CORe-clinical-diagnosis-prediction" #emilyalsentzer/Bio_ClinicalBERT