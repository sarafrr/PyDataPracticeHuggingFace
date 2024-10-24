from typing import Tuple
import os
from glob import glob
from tqdm import tqdm
from PIL import Image

import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
# nltk.download('words')
import Levenshtein

def get_partitioned_data(
    path_files : str,
    list_files_train: list, 
    list_files_val :list,
    list_files_test :list)->Tuple[Tuple[list,list], Tuple[list,list], Tuple[list,list]]:

    list_names_train, list_transcriptions_train = [], []
    list_names_val, list_transcriptions_val = [], []
    list_names_test, list_transcriptions_test = [], []
    for im in glob(os.path.join(path_files,'*.png')):
        name_img = im.split('/')[-1]
        name_file = os.path.splitext(name_img)[0]+'.txt'
        if name_img+'\n' in list_files_train:
            with open(os.path.join(path_files,name_file)) as t:
                text = t.readline()
                list_transcriptions_train.append(text)
                list_names_train.append(name_img)
        elif name_img+'\n' in list_files_val:
            with open(os.path.join(path_files,name_file)) as t:
                text = t.readline()
                list_transcriptions_val.append(text)
                list_names_val.append(name_img)
        elif name_img+'\n' in list_files_test:
            with open(os.path.join(path_files,name_file)) as t:
                text = t.readline()
                list_transcriptions_test.append(text)
                list_names_test.append(name_img)
    return (list_names_train, list_transcriptions_train), (list_names_val, list_transcriptions_val), (list_names_test, list_transcriptions_test)

def read_image(image_path):
    """
    :param image_path: String, path to the input image.
 
 
    Returns:
        image: PIL Image.
    """
    image = Image.open(image_path).convert('RGB')
    return image

def ocr(image, processor, model, device = 'cpu'):
    """
    :param image: PIL Image.
    :param processor: Huggingface OCR processor.
    :param model: Huggingface OCR model.
    :param device: the device to use
 
 
    Returns:
        generated_text: the OCR'd text string.
    """
    # We can directly perform OCR on cropped images.
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def get_list_transcriptions(path_to_dataset : str = './', split : str = 'train'):
    list_transcriptions = []
    with open(path_to_dataset[:-5]+split+'.ln', "r") as f:
        name_lines = f.readlines()
    for l in name_lines:
        with open(os.path.join(path_to_dataset,l.strip()[:-4]+'.txt'), 'r') as f1:
            list_transcriptions.append(f1.readline().strip())
    return list_transcriptions

def get_set_words(transcriptions : list):
    set_words = set()
    for l in transcriptions:
        words = l.split(' ')
        set_words |= set(words)
    return set_words


def most_similar_word_levenshtein(word, custom_set, th_lev_dist : float = 0.2):
    vocabulary = (words.words() or custom_set)
    if word == '':
        return word
    elif word in vocabulary:
        return word
    else:
        original = word
        if original.istitle():
            return original
        min_distance = float('inf')
        most_similar = word

        for vocab_word in custom_set:
            distance = Levenshtein.distance(word, vocab_word)
            if distance < min_distance:
                min_distance = distance
                most_similar = vocab_word
        for vocab_word in words.words():
            if word[-1]=='s':
                vocab_word = vocab_word + 's'
                distance = Levenshtein.distance(word, vocab_word)
                if distance < min_distance:
                    min_distance = distance
                    most_similar = vocab_word

        if min_distance/len(word) > th_lev_dist:
            return original
        return most_similar

def simple_htr_corr(
    list_htr_transcriptions : list, \
    custom_set : set = set(), \
    name_file : str = 'simple_corr_htr.txt')->list:

    post_processed_lines = []
    with open(name_file, 'w+') as f:
        i = 0
        for l in tqdm(list_htr_transcriptions):
            i += 1
            _words = l.strip('\n').split(' ')
            corrected_line = []
            for w in _words:
                out = most_similar_word_levenshtein(w, custom_set)
                corrected_line.append(out)
            out = ' '.join(corrected_line)
            post_processed_lines.append(out)
            f.write(out)
            f.write('\n')
    return post_processed_lines


def generate_masked_strings(
    input_string,
    mask_str : str = '<mask>'
    )->list:
    '''
        Generates all the possible string with one masked token

        Args
        ----
        :param str input_string: the input string to modify by masking a word
        :param str mask_str: the string to use to mask
        :return the list of strings with all the strings with one masked token
        
        Note
        ----
        BERT and RoBERTa have two syntaxes to specify the mask
        token:
        - BERT by '[MASK]'
        - RoBERTa by '<mask>'
    '''
    words = input_string.split()
    masked_strings = []

    for i, word in enumerate(words):
        masked_words = words.copy()
        masked_words[i] = mask_str
        masked_strings.append(' '.join(masked_words))

    return masked_strings

def generate_masked_err_strings(
    input_string,
    custom_set : set = {}, 
    use_external_eng_voc : bool = False,
    mask_str : str = '<mask>'
    )->list:
    '''
    This function can leverage on an external English dictionary.
    However, it is possible to pass a custom set of words

    Args
    ----
    :param str input_string: the input string to modify by masking a word
    :param set custom_set: to pass a custom set of words (e.g., the ones of training)
    :param bool use_external_eng_voc: if to use an English extenral vocabulary
    :param str mask_str: the string to use to mask
    :return the list of strings with all the strings with one masked token which is
        considered as erroneous, because it is not in the vocabulary
    
    Note
    ----
    BERT and RoBERTa have two syntaxes to specify the mask
    token:
    - BERT by '[MASK]'
    - RoBERTa by '<mask>'
    '''
    _words = input_string.split()
    masked_strings = []
    if use_external_eng_voc:
        used_dict = words.words() or custom_set
    else:
        used_dict = custom_set
    for i, w in enumerate(_words):
        masked_words = _words.copy()
        if w not in used_dict:
            masked_words[i] = mask_str
        masked_strings.append(' '.join(masked_words))
    return masked_strings