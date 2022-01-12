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

root_dir = '/content/drive/MyDrive/ocr/root_folder'
file_names = [fn for fn in os.listdir(root_dir)] 
print("Select a file from the given list. Input the corresponding number")

counttmp=0
for f in file_names:
  counttmp+=1
  print(str(counttmp)+". "+f)

innerfolder=""
try:
  chosenFileNum=int(input("\nSelect a number from above based on the folder you wanted to OCR:\n"))-1
  innerfolder=file_names[chosenFileNum]
  if(innerfolder.find(" ")!=-1):
    raise NameError("File name contains spaces")
except Exception as err:
  print("Error: {0}".format(err))
  sys.exit(1)

try:
  output_dir = input("Enter name of directory for OCR output:\n")

  if(output_dir.find(" ")!=-1):
    raise NameError("File name contains spaces")
except Exception as err:
  print("Error: {0}".format(err))
  sys.exit(1)

pytesseract.tesseract_cmd = r'/content/drive/MyDrive/ocr/ocr-eval-exec/bin'
tessdata_dir_config = r'--tessdata-dir "/content/drive/MyDrive/ocr/tessdata"'

languages=pytesseract.get_languages(config=tessdata_dir_config)
lcount=0 
tesslanglist={}
print(languages)
for l in languages:
    tesslanglist[lcount]=l
    lcount+=1
    print(str(lcount)+'. '+l)

linput=input("Choose the lang model for OCR from the above list: ")

if not (int(linput)-1) in tesslanglist:
  print("Not a correct option! Exiting program")
  sys.exit(1)

print("Selected language model "+tesslanglist[int(linput)-1])

final_language = tesslanglist[int(linput)-1]

os.environ["TESSDATA_PREFIX"] = '/content/drive/MyDrive/ocr/tessdata'

print('done')
os.mkdir(root_dir+'/'+output_dir)

if os.path.isdir(root_dir+'/'+innerfolder):
    print(innerfolder)
    os.chdir(root_dir+'/'+innerfolder)
    for pdffile in os.listdir(root_dir+'/'+innerfolder):
      print(pdffile)
      
      if pdffile.endswith('.pdf'):
        
        file_name=pdffile.replace(".pdf","")
        newdir=os.path.join(root_dir+'/'+output_dir+'/'+file_name)
        os.mkdir(newdir)
        oldfilepath=os.path.join(root_dir,innerfolder,pdffile)
        newpdffilepath=os.path.join(root_dir,innerfolder,newdir,pdffile)
        os.replace(oldfilepath,newpdffilepath)
        print("OCRing file: ",newpdffilepath)
        imagesFolder = newdir + "/page_images"
        outputfolder=newdir + "/output_files"
        os.mkdir(imagesFolder)
        os.mkdir(outputfolder)
        
        convert_from_path(
          newpdffilepath,
          output_folder=imagesFolder,
          paths_only=True,
          fmt='jpg',
          output_file="O",
          use_pdftocairo=True,
        )
        
        pytesseract.tesseract_cmd = r'/content/drive/MyDrive/ocr/ocr-eval-exec/bin'
        tessdata_dir_config = r'--tessdata-dir "/content/drive/MyDrive/ocr/tessdata"'
        for imfile in os.listdir(imagesFolder):
          print(imagesFolder + "/" + imfile)

          txt = pytesseract.image_to_string(imagesFolder + "/" + imfile,
                                      lang='eng',
                                      config='--oem 1 --psm 3')

          with open(outputfolder + '/' + imfile[:-3] + 'txt', 'w') as f:
            f.write(txt)

      
      elif pdffile.endswith('.jpg'):
        
        file_name=pdffile.replace(".jpg","")
        
        pytesseract.tesseract_cmd = r'/content/drive/MyDrive/ocr/ocr-eval-exec/bin'
        tessdata_dir_config = r'--tessdata-dir "/content/drive/MyDrive/ocr/tessdata"'
        
        print(file_name)

        txt = pytesseract.image_to_string(root_dir+'/'+innerfolder+'/'+pdffile,
                                      lang=final_language,
                                      config='--oem 1 --psm 3')

        with open(root_dir+'/'+output_dir + '/' + pdffile[:-3] + 'txt', 'w') as f:
            f.write(txt)

      print('done with '+pdffile)
