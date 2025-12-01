import random

import pandas as pd
from tqdm import tqdm


def random_deletion(words, p=0.2):
    """ Randomly delete words with probability p """
    if len(words) == 1:
        return words
    return [w for w in words if random.uniform(0, 1) > p]


def random_swap(words, n=1):
    """ Swap two words n times """
    words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return words


def random_insertion(words, n=1):
    """ Insert a random word from the sentence into a random position """
    words = words.copy()
    for _ in range(n):
        new_word = random.choice(words)
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, new_word)
    return words


def augment_text(text):
    words = text.split()
    if len(words) == 1:
        return text
    choice = random.choice(['delete', 'swap', 'insert'])

    if choice == 'delete':
        aug_words = random_deletion(words)
    elif choice == 'swap':
        aug_words = random_swap(words)
    elif choice == 'insert':
        aug_words = random_insertion(words)

    return " ".join(aug_words)

def augment_dataset(df:pd.DataFrame , text_col="tweet", label_col="label", frac=0.1):
    """
    Augment a fraction of dataset rows and return a bigger dataset
    """
    # sample rows for augmentation
    sampled = df.sample(frac=frac, random_state=42)

    augmented_texts = []
    augmented_labels = []

    for _, row in tqdm(sampled.iterrows()):
        aug_text = augment_text(row[text_col])  # from previous code
        augmented_texts.append(aug_text)
        augmented_labels.append(row[label_col])

    # make augmented dataframe
    df_aug = pd.DataFrame({text_col: augmented_texts, label_col: augmented_labels})

    # concatenate with original
    df_new = pd.concat([df, df_aug], ignore_index=True)
    return df_new
