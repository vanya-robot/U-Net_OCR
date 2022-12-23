### Russian language OCR built with U-Net models and CRNN (plus additional Faster-RCNN for russian passport recognition task).


Use by running pipeline_general.py (for general text recognition purposes)

or pipeline_passport.py (for Russian Federation passport recognition)

General system pipeline:
1. **U-Net** trained to segment lines of text from image (pic. 1).
2. **U-Net** trained to segment words from lines of text (pic. 2).
3. **CRNN** trained to recognize words (fonts, on which it was trained are located in *./data_generator/fonts/*).

![alt text](https://user-images.githubusercontent.com/114025176/209322269-b933ba8a-d969-4eae-a22c-0bfa9bfa92bf.png)
<p align="center">
<b>pic. 1 (Mask of lines in the image)</b>
</p>


<p align="center">
<img width="352" src="https://user-images.githubusercontent.com/114025176/209322980-28c76de4-54d2-4da7-a772-25ed19ad114a.png">
</p>
<p align="center">
<b>pic. 2 (Mask of words in line)</b>
</p>


<p align="center">
<img width="352" src="https://user-images.githubusercontent.com/114025176/209323932-c6ec3061-4248-46ad-a13e-af655b9b1b4a.jpg">
</p>
<p align="center">
<b>pic. 3 (Recognition of words by CRNN)</b>
</p>

### For better performance 
You should re-train CRNN with fonts that it will be recognizing in your task (add your fonts to data_generator/fonts/ and proceed to Russian_OCR_part_1.ipynb). 

### For Russian passport recognition
Faster-RCNN were added to detect area of image with First name, Last name and Patronymic (pic. 4).

<p align="center">
<img width="144" src="https://user-images.githubusercontent.com/114025176/209326169-9e238463-f152-457a-9bce-de7cb9ae9fc9.png">
</p>
<p align="center">
<b>pic. 4 (Passport recognition)</b>
</p>

Then General pipeline is used to recognize this as regular image.