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
PICKLE_DIR_IN_PT = os.path.join(PICKLE_FOLDER, "in_pt_embbedings") 
RESULT_FOLDER = os.path.join(DATA_ROOT, "result")

ATTRIBUTES = ["page","line","text", "CODE"]

MODEL_NAMES = {
    "model_1": "bvanaken/CORe-clinical-diagnosis-prediction", # Accuracy -> 37.8%
    "model_2": "emilyalsentzer/Bio_ClinicalBERT", # Accuracy -> 19.9%
    "model_3": "bvanaken/clinical-assertion-negation-bert", # Acuracy -> 20.2%
    "model_4": "beatrice-portelli/DiLBERT", # Accuracy -> 48.2%
    "model_5": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12" # Accuracy-> 15%

}

BERT_PRETRAINED = MODEL_NAMES["model_4"]

EXEC = "nphi"
HELPER_IN_PT_PATH = "./data/helper/sample # 1 IN PT.xlsx"