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
import cv2
import pytesseract
import os
from pdf2image import convert_from_path
import cv2
import sys
from pdfreader import SimplePDFViewer

#initialise input image
fnames = [fn for fn in os.listdir('.')]
fnames_= {}
for k,f in enumerate(fnames):
  print(str(k+1) + ". " + str(f))
  fnames_[str(k+1)] = str(f)
dir_no = int(input("Please enter the image directory number:\n"))
img_dir = fnames_[str(dir_no)]
print(dir_no, "-", img_dir)

#initialise language model 
input_lang = input("Please enter the language of your images. If more than one languages are to be used, please enter like so '"'san+eng'"'")

#initialise output directory 
try:
  output_dir = input("Please enter the name of the output directory:\n")
  if(output_dir.find(" ")!=-1):
    raise NameError("File name contains spaces")
except Exception as err:
  print("Error: {0}".format(err))
  sys.exit(1)
  
if not os.path.exists(output_dir):
  os.mkdir(output_dir)
  
# try:
#   os.mkdir(output_dir)
# except:
#   print("Output directory already exists, please enter the name of a new directory.") #TO DO: User should be able to enter existing dirs as well. 

ocr_agent = lp.TesseractAgent(languages=input_lang) #TO DO - Explore the layout part, how it can be added

if os.path.isdir(img_dir):
  print(img_dir)
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
        with open(newdir + "/output/" + img_[:-4] + '.txt', 'w') as f:
          f.write(res)
    elif img_file.endswith('.jpg') or img_file.endswith('.png') or img_file.endswith('.jpeg'):
      print("OCR-ing images...\n")
      image = cv2.imread(img_dir + "/" + img_file)
      res = ocr_agent.detect(image)
      with open(output_dir + '/' + img_file[:-4] + '.txt', 'w') as f:
        f.write(res)

  print("OCR is complete. Please find output in provided output directory.")
else:
  print("You have not entered a valid image directory.")
