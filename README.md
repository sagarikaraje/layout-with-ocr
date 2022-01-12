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
- Run lp_ocr script: 
5. python3 lp_ocr.py 
