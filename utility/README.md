## Printed Text Recogniton Dataset Creation 

### Step 1: Getting unique words and vocabulary from input corpus

This step will provide the vocabulary (set of all printed characters) for which the recognition model will be trained on. 
It will also a generate a file having the list of all words in the concerned langauge whioch would be forming the actual dataset.
Please add language argument to select the langauge among kannada, devanagari, tamil and telugu

```
python3 generate_words_and_vocab.py --text_dir input/ --language devanagari
```
In the above example all the txt files in input folder will be scanned for devanagari language words.
'vocab.txt' file will be generated having the vocabulary for the dataset.
'unique_words' file will enlist all the unique words that constitute the dataset.
These files will be useful for further steps.

### Step 2: Generating images from unique words

Run the following script from the same place where 'unique_words' file is generated to create images from the words.
After successful execution, the images folder will have the set of images and labels.json will have the corresponding ground truth.
Provide the images folder and labels.json as the dataset path for starting the training process. 

```
python3 generate_word_images.py 
```

#### NOTE
The script requires concerned language font (present in line 43) and Tesseract text2image utility inorder to generate the dataset properly.