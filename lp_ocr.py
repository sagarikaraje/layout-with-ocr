# -*- coding: utf-8 -*-

import layoutparser as lp
import pandas as pd
import numpy as np
import cv2
import os
try:
 from PIL import Image
except ImportError:
 import Image
import pytesseract
from pdf2image import convert_from_path
import sys
from pdfreader import SimplePDFViewer
import subprocess  

# Execute layout inference

from layout_inference import infer_layout 

#does the user want to use layout inference ? 
infer_flag = input("Do you wish to use Layout Inference? (yes or no)")

if(input('Do you wish to specify Tessdata directory? \nDefault: $(LOCAL)/share/tessdata \nEnter y for yes, n for no: ') == 'y'):
  os.environ["TESSDATA_PREFIX"] = input("Enter path to tessdata directory: \n")

#initialise language model 
input_lang = input("Please enter the language of your images. If more than one languages are to be used, please enter like so '"'san+eng'"'")

#initialise output directory 
try:
  output_dir = input("Directory for OCR output: \n")
  if(output_dir.find(" ")!=-1):
    raise NameError("File name contains spaces")
except Exception as err:
  print("Error: {0}".format(err))
  sys.exit(1)

if not os.path.exists(output_dir):
  os.mkdir(output_dir) 

ocr_agent = lp.TesseractAgent(languages=input_lang) 

if infer_flag == "no":
  img_dir = input("Enter the name of our image folder for OCR")
  if os.path.isdir(img_dir):
    for img_file in os.listdir(img_dir):
      if img_file.endswith('.pdf'):
        print("OCR-ing pdfs...\n")
        newdir = output_dir + "/" + img_file.replace(".pdf", "")
        os.mkdir(newdir)
        os.mkdir(newdir + "/page_images")
        os.mkdir(newdir + "/output")
        img_path= img_dir + "/" + img_file
        print("Converting to images...\n")
        convert_from_path(img_path,
            output_folder= newdir + "/page_images",
            paths_only=True,
            fmt='jpg',
            output_file="O",
            use_pdftocairo=True,
          )
        for img_ in os.listdir(newdir + "/page_images"):
          print(img_)
          image = cv2.imread(newdir + "/page_images/" + img_)
          res = ocr_agent.detect(image)
          with open(newdir + "/output/" + img_[:-4] + 'txt', 'w') as f:
            f.write(res)
      elif img_file.endswith('.jpg') or img_file.endswith('.png') or img_file.endswith('.jpeg'):
        print("OCR-ing images...\n")
        image = cv2.imread(img_dir + "/" + img_file)
        res = ocr_agent.detect(image)
        if img_file.endswith('.jpeg'):
          x = img_file[:-5]
        else:
          x = img_file[:-4]
        with open(output_dir + '/' + x + '.txt', 'w') as f:
          f.write(res)
  print("OCR is complete. Please find the output in the provided output directory.")

elif infer_flag == "yes":
  img, layout_info = infer_layout()
  #sorting layout_info by y_1 coordinate
  layout_info_sort = {k: v for k, v in sorted(layout_info.items(), key=lambda item: item[1][1], reverse=True)}
  with open(output_dir + '/output-ocr.txt', 'w') as f:
    for label, box in layout_info_sort.items():
      img_cropped = img.crop(box)
      res = ocr_agent.detect(img_cropped)
      f.write(res)
    f.close()

  print("OCR is complete. Please find the output in the provided output directory.")

else:
  print('Incorrect Input')
