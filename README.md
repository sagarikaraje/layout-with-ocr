# layout-with-ocr
To run lp_ocr.py, wrapper for Layout Parser OCR **for the first time**: 
- Create a venv and activate:  
1. virtualenv lp_ocr
2. source lp_ocr/bin/activate
- Install all packages in the environment: 
1. pip3 install -r requirements.txt
2. apt install tesseract-ocr
3. apt install libtesseract-dev
4. apt-get install poppler-utils
- For document layout detection + OCR of an image:   
python3 lp_ocr.py 
(select 'yes' when asked if layout detection should be applied) 
- For document layout analysis of an image: 
python3 layout_inference.py 
(will return an infered image with masks and a json file with layout data) 
- For OCR of a directory of images: 
python3 lp_ocr.py 
(select 'no' when asked if layout detection should be applied, and supply your input image directory) 

