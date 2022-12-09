import glob
import argparse
import re
from tqdm import tqdm

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_dir', help='directory containing the text files.',type=str)
    parser.add_argument('--language', help='Indian language for dataset generation', type=str)
    args = parser.parse_args()
    
    if args.text_dir is None:
        print("Please add text_dir argument to point to a corpus folder having text files")
        exit()
    
    if args.language is None:
        print("Please add language argument to select the langauge among kannada, devanagari, tamil and telugu")
        exit()
        

    text_files = glob.glob(args.text_dir+'/*.txt')
    language_regex = '\u0900-\u097F'
    #default is devanagari
    if args.language == "kannada":
        language_regex = '\u0C80-\u0CFF'
    if args.language == "tamil":
        language_regex = '\u0B80-\u0BFF'
    if args.language == "telugu":
        language_regex = '\u0C00-\u0C7F'
    san_re=re.compile('[' + language_regex + ']+')
    #kannada: 0c80 to 0cff
    #devanagari: 0900 to 097F
    #tamil: 0B80 to 0BFF
    #telugu: 0C00 to 0C7F
            
    unique_words=[]
    characters = set()
    for text in tqdm(text_files):
        with open(text,'r') as f:
            for line in f:
                words=line.split(' ')
                for word in words:
                    m=san_re.match(word)
                    if m:
                        tword=m.group()
                        if len(tword) > 32:
                            continue
                        if tword not in unique_words:
                            unique_words.append(tword)
                            for c in tword:
                                characters.add(c)
    
    with open('vocab.txt','w') as f:
        f.write("".join(characters))

    with open('unique_words','w') as f:
        f.write("\n".join(unique_words))

    print("Unique words number is %d" % len(unique_words))  

if __name__ == '__main__':
    args()
