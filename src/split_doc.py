#File Splitting - Doc pre-processing
import logging as logger
logger.getLogger().setLevel(logger.INFO)

import os
from pdf2image import convert_from_path, pdfinfo_from_path

from config import doc_name, FILENAME, PROCESSED_FOLDER

from utils import rename_images

def split_doc_into_pages():
    logger.info(f"Splitting pdf {doc_name} into PNGs")
    output_folder = os.path.join(PROCESSED_FOLDER, doc_name.split(".")[0])
    os.makedirs(output_folder, exist_ok=True)
    num_pages = pdfinfo_from_path(FILENAME)['Pages']
    logger.info(
        f"Found {num_pages} pages of {doc_name} to split into PNGs")

    image_paths = convert_from_path(FILENAME,
                                    output_folder=output_folder,
                                    paths_only=True,
                                    fmt="png",
                                    output_file="page",
                                    )
    for image_path in image_paths:
        rename_images(image_path, output_folder)