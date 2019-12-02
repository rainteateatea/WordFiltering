import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import math
import matplotlib.pyplot as plt

# IO = 'hn2018_2019.csv'
IO = '100test.csv'
pd.set_option('max_colwidth', 1000)
# read bi-gram word library
# Bi_CSV = 'bigram_f.csv'
Bi_CSV = '9.csv'
bi_CSV = pd.read_csv(Bi_CSV)
bi_CSV.columns = ['word']
bi_words = bi_CSV['word']

remove_word = []
task = 1


def main():
    # process four classes : story, ask-hn, show-hn and poll
    df = pd.read_csv(IO)

    df['Created At'] = pd.to_datetime(df['Created At'])
    # story training dataset
    print('process story training dataset')
    training_story_data = df[(df['Created At'].dt.year.isin([2018])) & (df['Post Type'] == 'story')]
    story_list = processing_data_set(training_story_data['Title'])
    story_Series = pd.Series(story_list)
    story_freq = story_Series.value_counts()

    # ask_hn training dataset
    print('process ask_hn training dataset')
    training_ask_hn_data = df[(df['Created At'].dt.year.isin([2018])) & (df['Post Type'] == 'ask_hn')]
    ask_hn_list = processing_data_set(training_ask_hn_data['Title'])
    ask_hn_Series = pd.Series(ask_hn_list)
    ask_hn_freq = ask_hn_Series.value_counts()

    # show_hn training dataset
    print('process show_hn training dataset')
    training_show_hn_data = df[(df['Created At'].dt.year.isin([2018])) & (df['Post Type'] == 'show_hn')]
    show_hn_list = processing_data_set(training_show_hn_data['Title'])
    show_hn_Series = pd.Series(show_hn_list)
    show_hn_freq = show_hn_Series.value_counts()

    # poll training dataset
    print('processing poll training dataset')
    training_poll_hn_data = df[(df['Created At'].dt.year.isin([2018])) & (df['Post Type'] == 'poll')]
    poll_list = processing_data_set(training_poll_hn_data['Title'])
    poll_Series = pd.Series(poll_list)
    poll_freq = poll_Series.value_counts()

    # Create Vocabulary list
    print('process vocabulary list')
    vocabulary_freq, vocabulary = Create_vocabulary(story_list, ask_hn_list, show_hn_list, poll_list)

    # save vocabulary.txt
    print('save vocabulary.txt')
    save_txt_file('vocabulary.txt', vocabulary)

    # save remove_word.txt
    remove_Series = pd.Series(remove_word)
    remove_freq = remove_Series.value_counts()
    save_txt_file('remove_word.txt', list(remove_freq.index.values))
    task = 2

    # build training model
    print('build training model')
    story_dict, ask_hn_dict, show_hn_dict, poll_dict = compute_probability(vocabulary, story_freq, ask_hn_freq
                                                                           , show_hn_freq, poll_freq, 0.5, 1)
    # # task2 baseline_result.txt file
    # initial score
    story_train = training_story_data.shape[0]
    ask_hn_train = training_ask_hn_data.shape[0]
    show_hn_train = training_show_hn_data.shape[0]
    poll_train = training_poll_hn_data.shape[0]

    testing_data = df[df['Created At'].dt.year.isin([2019])]
    print('baseline-result')
    testing_dataset(testing_data, story_train, ask_hn_train, show_hn_train, poll_train,
                    story_dict, ask_hn_dict, show_hn_dict, poll_dict, 1)

    # stop-word filtering
    print('Experiment 2: Stop-word Filtering')
    stop_dict = read_stop_words()
    new_vocabulary = stop_word_filtering(stop_dict, vocabulary)
    print('build stopword model')
    story_dict, ask_hn_dict, show_hn_dict, poll_dict = compute_probability(new_vocabulary, story_freq, ask_hn_freq
                                                                           , show_hn_freq, poll_freq, 0.5, 2)
    print('stopword-result')
    testing_dataset(testing_data, story_train, ask_hn_train, show_hn_train, poll_train,
                    story_dict, ask_hn_dict, show_hn_dict, poll_dict, 2)
    # word length filtering
    print('Experiment 3: word length filtering')
    new_vocabulary = word_length_filtering(vocabulary)
    print('build wordlength-model')
    story_dict, ask_hn_dict, show_hn_dict, poll_dict = compute_probability(new_vocabulary, story_freq, ask_hn_freq
                                                                           , show_hn_freq, poll_freq, 0.5, 3)
    print('wordlength-result')
    testing_dataset(testing_data, story_train, ask_hn_train, show_hn_train, poll_train,
                    story_dict, ask_hn_dict, show_hn_dict, poll_dict, 3)
    # Infrequent Word Filtering
    print('Experiment 4: Infrequent Word Filtering')
    # new_vocabulary = word_length_filtering(1, vocabulary_freq)
    start_word_filtering(vocabulary_freq, story_freq, ask_hn_freq, show_hn_freq, poll_freq, testing_data, story_train,
                         ask_hn_train, show_hn_train, poll_train)
    print('Experiment 5: Smoothing')
    accuracy_int_list = []

    precision_story_int_list = []
    precision_ask_hn_int_list = []
    precision_show_hn_int_list = []
    precision_poll_int_list = []

    recall_story_int_list = []
    recall_ask_hn_int_list = []
    recall_show_hn_int_list = []
    recall_poll_int_list = []

    f_measure_story_int_list = []
    f_measure_ask_hn_int_list = []
    f_measure_show_hn_int_list = []
    f_measure_poll_int_list = []
    # smoothing
    smooth = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for sm in smooth:
        print()
        print('smooth = ' + str(sm))

        # build smooth model
        story_dict, ask_hn_dict, show_hn_dict, poll_dict = compute_probability(vocabulary, story_freq, ask_hn_freq
                                                                               , show_hn_freq, poll_freq, sm, 5)
        # build result
        accuracy, precision_story, recall_story, f_story, precision_ask_hn, recall_ask_hn, f_ask_hn, precision_show_hn, \
        recall_show_hn, f_show_hn, precision_poll, recall_poll, f_poll = testing_dataset(testing_data, story_train,
                                                                                         ask_hn_train, show_hn_train,
                                                                                         poll_train, story_dict,
                                                                                         ask_hn_dict, show_hn_dict,
                                                                                         poll_dict, 4)
        accuracy_int_list.append(accuracy)

        precision_story_int_list.append(precision_story)
        recall_story_int_list.append(recall_story)
        f_measure_story_int_list.append(f_story)

        precision_ask_hn_int_list.append(precision_ask_hn)
        recall_ask_hn_int_list.append(recall_ask_hn)
        f_measure_ask_hn_int_list.append(f_ask_hn)

        precision_show_hn_int_list.append(precision_show_hn)
        recall_show_hn_int_list.append(recall_show_hn)
        f_measure_show_hn_int_list.append(f_show_hn)

        precision_poll_int_list.append(precision_poll)
        recall_poll_int_list.append(recall_poll)
        f_measure_poll_int_list.append(f_poll)
        print()
    smooth_figure(smooth, accuracy_int_list, precision_story_int_list, recall_story_int_list, f_measure_story_int_list,
                  precision_ask_hn_int_list, recall_ask_hn_int_list, f_measure_ask_hn_int_list,
                  precision_show_hn_int_list, recall_show_hn_int_list, f_measure_show_hn_int_list,
                  precision_poll_int_list, recall_poll_int_list, f_measure_poll_int_list)


def smooth_figure(smooth, accuracy, precision_story, recall_story, f_measure_story,
                  precision_ask_hn, recall_ask_hn, f_measure_ask_hn,
                  precision_show_hn, recall_show_hn, f_measure_show_hn,
                  precision_poll, recall_poll, f_measure_poll):
    x = range(len(smooth))
    # accuracy figure
    plt.figure(1)
    plt.title('smooth accuracy')
    plt.plot(x, accuracy)
    plt.xticks(x, smooth)
    plt.ylabel('Accuracy')
    plt.xlabel('the number of words left in your vocabulary')
    plt.savefig('Accuracy_smooth.png')
    plt.close()
    # precision figure
    plt.figure(2)
    l1 = plt.plot(x, precision_story, 'r--', label='story')
    l2 = plt.plot(x, precision_ask_hn, 'g--', label='ask_hn')
    l3 = plt.plot(x, precision_show_hn, 'b--', label='show_hn')
    l4 = plt.plot(x, precision_poll, 'y--', label='poll')
    plt.legend()
    plt.xticks(x, smooth)
    plt.title('smooth precision')
    plt.xlabel('smooth value from 0 to 1')
    plt.ylabel('Precision')
    plt.savefig('Precision_smooth.png')
    plt.close()
    # recall figure
    plt.figure(3)
    l1 = plt.plot(x, recall_story, 'r--', label='story')
    l2 = plt.plot(x, recall_ask_hn, 'g--', label='ask_hn')
    l3 = plt.plot(x, recall_show_hn, 'b--', label='show_hn')
    l4 = plt.plot(x, recall_poll, 'y--', label='poll')
    plt.legend()
    plt.xticks(x, smooth)
    plt.title('smooth recall')
    plt.xlabel('the number of words left in your vocabulary')
    plt.ylabel('Recall')
    plt.savefig('Recall_smooth.png')
    plt.close()

    # f-measure
    plt.figure(4)
    l1 = plt.plot(x, f_measure_story, 'r--', label='story')
    l2 = plt.plot(x, f_measure_ask_hn, 'g--', label='ask_hn')
    l3 = plt.plot(x, f_measure_show_hn, 'b--', label='show_hn')
    l4 = plt.plot(x, f_measure_poll, 'y--', label='poll')
    plt.legend()
    plt.xticks(x, smooth)
    plt.title('smooth F-measure')
    plt.xlabel('the number of words left in your vocabulary')
    plt.ylabel('F-measure')
    plt.savefig('F_measure_smooth.png')
    plt.close()


def start_word_filtering(vocabulary_freq, story_freq, ask_hn_freq, show_hn_freq, poll_freq, testing_data, story_train,
                         ask_hn_train, show_hn_train, poll_train):
    int_vocab = []
    float_vocab = []
    accuracy_int_list = []

    precision_story_int_list = []
    precision_ask_hn_int_list = []
    precision_show_hn_int_list = []
    precision_poll_int_list = []

    recall_story_int_list = []
    recall_ask_hn_int_list = []
    recall_show_hn_int_list = []
    recall_poll_int_list = []

    f_measure_story_int_list = []
    f_measure_ask_hn_int_list = []
    f_measure_show_hn_int_list = []
    f_measure_poll_int_list = []

    accuracy_float_list = []

    recall_story_float_list = []
    recall_ask_hn_float_list = []
    recall_show_hn_float_list = []
    recall_poll_float_list = []

    precision_story_float_list = []
    precision_ask_hn_float_list = []
    precision_show_hn_float_list = []
    precision_poll_float_list = []

    f_measure_story_float_list = []
    f_measure_ask_hn_float_list = []
    f_measure_show_hn_float_list = []
    f_measure_poll_float_list = []
    # frequency = 1, <=5, <=10, <=15, <=20
    freq_list = [1, 5, 10, 15, 20, 0.01, 0.05, 0.1,0.15,0.2]
    for f in freq_list:
        print('remove from the vocabulary words with frequency = ' + str(f))
        new_vocabulary = infrequent_word_filtering(f, vocabulary_freq)
        # build model
        story_dict, ask_hn_dict, show_hn_dict, poll_dict = compute_probability(new_vocabulary, story_freq, ask_hn_freq
                                                                               , show_hn_freq, poll_freq, 0.5, 4)
        # build result
        accuracy, precision_story, recall_story, f_story, precision_ask_hn, recall_ask_hn, f_ask_hn, precision_show_hn, \
        recall_show_hn, f_show_hn, precision_poll, recall_poll, f_poll = testing_dataset(testing_data, story_train,
                                                                                         ask_hn_train, show_hn_train,
                                                                                         poll_train, story_dict,
                                                                                         ask_hn_dict, show_hn_dict,
                                                                                         poll_dict, 4)
        if f >= 1:
            int_vocab.append(len(new_vocabulary))

            accuracy_int_list.append(accuracy)

            precision_story_int_list.append(precision_story)
            recall_story_int_list.append(recall_story)
            f_measure_story_int_list.append(f_story)

            precision_ask_hn_int_list.append(precision_ask_hn)
            recall_ask_hn_int_list.append(recall_ask_hn)
            f_measure_ask_hn_int_list.append(f_ask_hn)

            precision_show_hn_int_list.append(precision_show_hn)
            recall_show_hn_int_list.append(recall_show_hn)
            f_measure_show_hn_int_list.append(f_show_hn)

            precision_poll_int_list.append(precision_poll)
            recall_poll_int_list.append(recall_poll)
            f_measure_poll_int_list.append(f_poll)
        else:
            float_vocab.append(len(new_vocabulary))

            accuracy_float_list.append(accuracy)

            precision_story_float_list.append(precision_story)
            recall_story_float_list.append(recall_story)
            f_measure_story_float_list.append(f_story)

            precision_ask_hn_float_list.append(precision_ask_hn)
            recall_ask_hn_float_list.append(recall_ask_hn)
            f_measure_ask_hn_float_list.append(f_ask_hn)

            precision_show_hn_float_list.append(precision_show_hn)
            recall_show_hn_float_list.append(recall_show_hn)
            f_measure_show_hn_float_list.append(f_show_hn)

            precision_poll_float_list.append(precision_poll)
            recall_poll_float_list.append(recall_poll)
            f_measure_poll_float_list.append(f_poll)

    # show figure 1 5 10
    show_figure(int_vocab, accuracy_int_list, precision_story_int_list, recall_story_int_list, f_measure_story_int_list,
                precision_ask_hn_int_list, recall_ask_hn_int_list, f_measure_ask_hn_int_list,
                precision_show_hn_int_list, recall_show_hn_int_list, f_measure_show_hn_int_list,
                precision_poll_int_list, recall_poll_int_list, f_measure_poll_int_list)
    # 5% 10%
    show_figure2(float_vocab, accuracy_float_list, precision_story_float_list, recall_story_float_list,
                 f_measure_story_float_list,
                 precision_ask_hn_float_list, recall_ask_hn_float_list, f_measure_ask_hn_float_list,
                 precision_show_hn_float_list, recall_show_hn_float_list, f_measure_show_hn_float_list,
                 precision_poll_float_list, recall_poll_float_list, f_measure_poll_float_list)


# 1% 5% 10% 15% 20% 
def show_figure2(vocab, accuracy, precision_story, recall_story, f_measure_story,
                 precision_ask_hn, recall_ask_hn, f_measure_ask_hn,
                 precision_show_hn, recall_show_hn, f_measure_show_hn,
                 precision_poll, recall_poll, f_measure_poll):
    x = range(5)
    # accuracy figure
    plt.figure(1)
    plt.title('Infrequent Word Filtering 1% 5% 10% 15% 20% ')
    plt.plot(x, accuracy)
    plt.xticks(x, vocab)
    plt.ylabel('Accuracy')
    plt.xlabel('the number of words left in your vocabulary')
    plt.savefig('Accuracy_float.png')
    plt.close()
    # precision figure
    plt.figure(2)
    l1 = plt.plot(x, precision_story, 'r--', label='story')
    l2 = plt.plot(x, precision_ask_hn, 'g--', label='ask_hn')
    l3 = plt.plot(x, precision_show_hn, 'b--', label='show_hn')
    l4 = plt.plot(x, precision_poll, 'y--', label='poll')
    plt.title('precision 1% 5% 10% 15% 20%')
    plt.legend()
    plt.xticks(x, vocab)
    plt.xlabel('the number of words left in your vocabulary')
    plt.ylabel('Precision')
    plt.savefig('Precision_float.png')
    plt.close()
    # recall figure
    plt.figure(3)
    l1 = plt.plot(x, recall_story, 'r--', label='story')
    l2 = plt.plot(x, recall_ask_hn, 'g--', label='ask_hn')
    l3 = plt.plot(x, recall_show_hn, 'b--', label='show_hn')
    l4 = plt.plot(x, recall_poll, 'y--', label='poll')
    plt.legend()
    plt.xticks(x, vocab)
    plt.title('Recall 1% 5% 10% 15% 20%')
    plt.xlabel('the number of words left in your vocabulary')
    plt.ylabel('Recall')
    plt.savefig('Recall_float.png')
    plt.close()

    # f-measure
    plt.figure(4)
    l1 = plt.plot(x, f_measure_story, 'r--', label='story')
    l2 = plt.plot(x, f_measure_ask_hn, 'g--', label='ask_hn')
    l3 = plt.plot(x, f_measure_show_hn, 'b--', label='show_hn')
    l4 = plt.plot(x, f_measure_poll, 'y--', label='poll')
    plt.legend()
    plt.xticks(x, vocab)
    plt.title('f-measure')
    plt.xlabel('the number of words left in your vocabulary')
    plt.ylabel('F-measure')
    plt.savefig('F_measure_float.png')
    plt.close()

# 1 5 10 15 20
def show_figure(vocab, accuracy, precision_story, recall_story, f_measure_story,
                precision_ask_hn, recall_ask_hn, f_measure_ask_hn,
                precision_show_hn, recall_show_hn, f_measure_show_hn,
                precision_poll, recall_poll, f_measure_poll):
    x = range(5)
    # accuracy figure
    plt.figure(1)
    plt.title('Infrequent Word Filtering <=1 5 10 15 20')
    plt.plot(x, accuracy)
    plt.xticks(x, vocab)
    plt.ylabel('Accuracy')
    plt.xlabel('the number of words left in your vocabulary')
    plt.savefig('Accuracy_int.png')
    plt.close()
    # precision figure
    plt.figure(2)
    l1 = plt.plot(x, precision_story, 'r--', label='story')
    l2 = plt.plot(x, precision_ask_hn, 'g--', label='ask_hn')
    l3 = plt.plot(x, precision_show_hn, 'b--', label='show_hn')
    l4 = plt.plot(x, precision_poll, 'y--', label='poll')
    plt.legend()
    plt.xticks(x, vocab)
    plt.title('precision <=1 5 10 15 20')
    plt.xlabel('the number of words left in your vocabulary')
    plt.ylabel('Precision')
    plt.savefig('Precision_int.png')
    plt.close()
    # recall figure
    plt.figure(3)
    l1 = plt.plot(x, recall_story, 'r--', label='story')
    l2 = plt.plot(x, recall_ask_hn, 'g--', label='ask_hn')
    l3 = plt.plot(x, recall_show_hn, 'b--', label='show_hn')
    l4 = plt.plot(x, recall_poll, 'y--', label='poll')
    plt.legend()
    plt.xticks(x, vocab)
    plt.title('recall <=1 5 10 15 20')
    plt.xlabel('the number of words left in your vocabulary')
    plt.ylabel('Recall')
    plt.savefig('Recall_int.png')
    plt.close()

    # f-measure
    plt.figure(4)
    l1 = plt.plot(x, f_measure_story, 'r--', label='story')
    l2 = plt.plot(x, f_measure_ask_hn, 'g--', label='ask_hn')
    l3 = plt.plot(x, f_measure_show_hn, 'b--', label='show_hn')
    l4 = plt.plot(x, f_measure_poll, 'y--', label='poll')
    plt.legend()
    plt.xticks(x, vocab)
    plt.title('f-measure')
    plt.xlabel('the number of words left in your vocabulary')
    plt.ylabel('F-measure')
    plt.savefig('f_measure_int.png')
    plt.close()


def infrequent_word_filtering(frequency, vocabulary):
    new_vocabulary = []
    if frequency >= 1:
        for word in vocabulary.iteritems():
            if word[1] > frequency:
                new_vocabulary.append(word[0])
    # remove top 5% 10% 15% 20% 25%
    else:
        total_vocab = vocabulary.shape[0]
        top = total_vocab * frequency
        i = 1
        for word in vocabulary.iteritems():
            if i > top:
                new_vocabulary.append(word[0])
            i = i + 1

    return new_vocabulary


def word_length_filtering(vocabulary):
    new_vocabulary = []
    # remove all words with length <= 2 and all words with length >=9
    for word in vocabulary:
        if 2 < len(word) < 9:
            new_vocabulary.append(word)
    return new_vocabulary


def stop_word_filtering(stop_dict, vocabulary):
    new_vocabulary = []
    for word in vocabulary:
        if word not in stop_dict.keys():
            new_vocabulary.append(word)
    return new_vocabulary


def read_stop_words():
    stop_dict = {}
    f = open('Stopwords.txt', 'r')
    for line in f.readlines():
        line = line.strip('\n')
        stop_dict[line] = 0
    return stop_dict


def testing_dataset(testing_data, story_train, ask_hn_train, show_hn_train, poll_train,
                    story_dict, ask_hn_dict, show_hn_dict, poll_dict, figure):
    result = []
    total_dataset = story_train + ask_hn_train + show_hn_train + poll_train

    story_score_init = math.log10(story_train / total_dataset)
    ask_hn_score_init = math.log10(ask_hn_train / total_dataset)
    show_hn_score_init = math.log10(show_hn_train / total_dataset)
    poll_score_init = math.log10(poll_train / total_dataset)

    # instance = target
    story_rignt = 0
    ask_hn_right = 0
    show_hn_right = 0
    poll_right = 0

    # instance number
    story_number = 0
    ask_hn_number = 0
    show_hn_number = 0
    poll_number = 0

    # instance number in testing
    story_test = 0
    ask_hn_test = 0
    show_hn_test = 0
    poll_test = 0

    i = 1
    # testing data
    for index, v in testing_data[['Title', 'Post Type']].iterrows():
        # analyze sentence return word list
        words = filter_character(v['Title'])
        target = v['Post Type']
        # score
        story_score = story_score_init
        ask_hn_score = ask_hn_score_init
        show_hn_score = show_hn_score_init
        poll_score = poll_score_init
        for word in words:
            # story
            if word in story_dict.keys():
                story_score = story_score + story_dict[word][1]
            # ask_hn
            if word in ask_hn_dict.keys():
                ask_hn_score = ask_hn_score + ask_hn_dict[word][1]
            # show_hn
            if word in show_hn_dict.keys():
                show_hn_score = show_hn_score + show_hn_dict[word][1]
            # poll
            if word in poll_dict.keys():
                poll_score = poll_score + poll_dict[word][1]
        # find max score
        score_dict = {'story': story_score, 'ask_hn': ask_hn_score, 'show_hn': show_hn_score, 'poll': poll_score}
        max_score = max(score_dict, key=score_dict.get)
        right_or_wrong = ''
        if target == 'story':
            story_test = story_test + 1
            if max_score == 'story':
                story_rignt = story_rignt + 1
                story_number = story_number + 1
                right_or_wrong = 'right'
            elif max_score == 'ask_hn':
                ask_hn_number = ask_hn_number + 1
                right_or_wrong = 'wrong'
            elif max_score == 'show_hn':
                show_hn_number = show_hn_number + 1
                right_or_wrong = 'wrong'
            elif max_score == 'poll':
                poll_number = poll_number + 1
                right_or_wrong = 'wrong'
            else:
                print('error')
        elif target == 'ask_hn':
            ask_hn_test = ask_hn_test + 1
            if max_score == 'story':
                right_or_wrong = 'wrong'
                story_number = story_number + 1
            elif max_score == 'ask_hn':
                right_or_wrong = 'right'
                ask_hn_right = ask_hn_right + 1
                ask_hn_number = ask_hn_number + 1
            elif max_score == 'show_hn':
                right_or_wrong = 'wrong'
                show_hn_number = show_hn_number + 1
            elif max_score == 'poll':
                right_or_wrong = 'wrong'
                poll_number = poll_number + 1
            else:
                print('error')
        elif target == 'show_hn':
            show_hn_test = show_hn_test + 1
            if max_score == 'story':
                right_or_wrong = 'wrong'
                story_number = story_number + 1
            elif max_score == 'ask_hn':
                right_or_wrong = 'wrong'
                ask_hn_number = ask_hn_number + 1
            elif max_score == 'show_hn':
                right_or_wrong = 'right'
                show_hn_right = show_hn_right + 1
                show_hn_number = show_hn_number + 1
            elif max_score == 'poll':
                right_or_wrong = 'wrong'
                poll_number = poll_number + 1
            else:
                print('error')
        elif target == 'poll':
            poll_test = poll_test + 1
            if max_score == 'story':
                right_or_wrong = 'wrong'
                story_number = story_number + 1
            elif max_score == 'ask_hn':
                right_or_wrong = 'wrong'
                ask_hn_number = ask_hn_number + 1
            elif max_score == 'show_hn':
                right_or_wrong = 'wrong'
                show_hn_number = show_hn_number + 1
            elif max_score == 'poll':
                right_or_wrong = 'right'
                poll_right = poll_right + 1
                poll_number = poll_number + 1
            else:
                print('error')

        i = i + 1
        line = str(i) + '  ' + v['Title'] + '  ' + max_score + '  ' + str(story_score) \
               + '  ' + str(ask_hn_score) + '  ' + str(show_hn_score) + '  ' + str(
            poll_score) + '  ' + target + '  ' + right_or_wrong
        result.append(line)

    # compute Accuracy recall precision f-measure
    total_test = story_test + ask_hn_test + show_hn_test + poll_test
    accuracy = (story_rignt + ask_hn_right + show_hn_right + poll_right) / total_test
    # precision
    precision_story = precision_recall(story_rignt, story_number)
    precision_ask_hn = precision_recall(ask_hn_right, ask_hn_number)
    precision_show_hn = precision_recall(show_hn_right, show_hn_number)
    precision_poll = precision_recall(poll_right, poll_number)
    # recall
    recall_story = precision_recall(story_rignt, story_test)
    recall_ask_hn = precision_recall(ask_hn_right, ask_hn_test)
    recall_show_hn = precision_recall(show_hn_right, show_hn_test)
    recall_poll = precision_recall(poll_right, poll_test)
    # f-measure
    f_story = f_measure(precision_story, recall_story)
    f_ask_hn = f_measure(precision_ask_hn, recall_ask_hn)
    f_show_hn = f_measure(precision_show_hn, recall_show_hn)
    f_poll = f_measure(precision_poll, recall_poll)
    filename = ''
    if figure == 1:
        # print baseline-result.txt
        baseline_result('baseline-result.txt', result)
        filename = 'baseline_result'
    elif figure == 2:
        # print stopword-result.txt
        baseline_result('stopword-result.txt', result)
        filename = 'stopword-result'
    elif figure == 3:
        baseline_result('wordlength-result.txt', result)
        filename = 'wordlength-result'

    baseline = filename + ': Accuracy : ' + str(accuracy)
    baseline_story = 'Precision_story: ' + str(precision_story) \
                     + ' Recall story: ' + str(recall_story) + 'F-measure story: ' + str(f_story)
    baseline_ask_hn = ' Precision_ask hn: ' + str(precision_ask_hn) + ' Recall ask hn: ' \
                      + str(recall_ask_hn) + ' F-measure ask hn: ' + str(f_ask_hn)
    baseline_show_hn = 'Precision show hn: ' + str(precision_show_hn) + ' Recall show hn: ' \
                       + str(recall_show_hn) + ' F-measure show hn: ' + str(f_show_hn)
    baseline_poll = 'Precision poll: ' + str(precision_poll) + ' Recall poll: ' \
                    + str(recall_poll) + ' F-measure poll: ' + str(f_poll)
    print('story_test ' + str(story_test) + ' ask_hn_test' + str(ask_hn_test)
          + ' show_hn_test ' + str(show_hn_test) + ' poll_test ' + str(poll_test))
    print('story_right '+str(story_rignt) + ' ask_hn right '+str(ask_hn_right) + ' show_hn_right '
          + str(show_hn_right) + ' poll_right ' + str(poll_right))
    print('story_number ' + str(story_number) + ' ask_hn_number ' + str(ask_hn_number)
          + ' show_hn_number ' + str(show_hn_number) + ' poll_number ' + str(poll_number))

    print(baseline)
    print(baseline_story)
    print(baseline_ask_hn)
    print(baseline_show_hn)
    print(baseline_poll)
    if figure == 4:
        return accuracy, precision_story, recall_story, f_story, precision_ask_hn, recall_ask_hn, f_ask_hn, \
               precision_show_hn, recall_show_hn, f_show_hn, precision_poll, recall_poll, f_poll


def precision_recall(instance_right, denominator):
    try:
        x = instance_right / denominator
    except ZeroDivisionError:
        x = 0
    return x


def f_measure(precision, recall):
    try:
        x = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        x = 0
    return x


def baseline_result(filename, result):
    file = open(filename, "w", encoding='utf-8')
    for line in result:
        file.write(line + '\n')
    file.close()


def compute_probability(vocabulary, story_freq, ask_hn_freq, show_hn_freq, poll_freq, smooth, task):
    total_story = np.sum(np.asarray(story_freq.values))
    total_ask_hn = np.sum(np.asarray(ask_hn_freq.values))
    total_show_hn = np.sum(np.asarray(show_hn_freq.values))
    total_poll = np.sum(np.asarray(poll_freq.values))
    vocabulary_size = len(vocabulary)

    story_dict = {}
    ask_hn_dict = {}
    show_hn_dict = {}
    poll_dict = {}
    for wi in vocabulary:
        # story smooth != 0
        if not smooth == 0:
            # word exist
            if wi in story_freq:
                P_wi_story = math.log10((story_freq[wi] + smooth) / (total_story + vocabulary_size))
                story_dict[wi] = [story_freq[wi], P_wi_story]
            # word not exist
            else:
                P_wi_story = math.log10(smooth / (total_story + vocabulary_size))
                story_dict[wi] = [0, P_wi_story]
        # story smooth ==0
        else:
            # word exist
            if wi in story_freq:
                P_wi_story = math.log10((story_freq[wi] + smooth) / (total_story + vocabulary_size))
                story_dict[wi] = [story_freq[wi], P_wi_story]
            # not exist
            else:
                P_wi_story = -float('inf')
                story_dict[wi] = [0, P_wi_story]

        # ask_hn
        if not smooth == 0:
            # word exist
            if wi in ask_hn_freq:
                P_wi_ask_hn = math.log10((ask_hn_freq[wi] + smooth) / (total_ask_hn + vocabulary_size))
                ask_hn_dict[wi] = [ask_hn_freq[wi], P_wi_ask_hn]
            # word not exist
            else:
                P_wi_ask_hn = math.log10(smooth / (total_ask_hn + vocabulary_size))
                ask_hn_dict[wi] = [0, P_wi_ask_hn]
        # ask_hn smooth == 0
        else:
            # word exist
            if wi in ask_hn_freq:
                P_wi_ask_hn = math.log10((ask_hn_freq[wi] + smooth) / (total_ask_hn + vocabulary_size))
                ask_hn_dict[wi] = [ask_hn_freq[wi], P_wi_ask_hn]
            # word not exist
            else:
                P_wi_ask_hn = -float('inf')
                ask_hn_dict[wi] = [0, P_wi_ask_hn]

        # show_hn
        # smooth != 0
        if not smooth == 0:
            if wi in show_hn_freq:
                P_wi_show_hn = math.log10((show_hn_freq[wi] + smooth) / (total_show_hn + vocabulary_size))
                show_hn_dict[wi] = [show_hn_freq[wi], P_wi_show_hn]
            else:
                P_wi_show_hn = math.log10(smooth / (total_show_hn + vocabulary_size))
                show_hn_dict[wi] = [0, P_wi_show_hn]
        # show_hn smooth ==0
        else:
            if wi in show_hn_freq:
                P_wi_show_hn = math.log10((show_hn_freq[wi] + smooth) / (total_show_hn + vocabulary_size))
                show_hn_dict[wi] = [show_hn_freq[wi], P_wi_show_hn]
            else:
                P_wi_show_hn = -float('inf')
                show_hn_dict[wi] = [0, P_wi_show_hn]
        # poll
        # smooth !=0
        if not smooth == 0:
            # word exists
            if wi in poll_freq:
                P_wi_poll = math.log10(poll_freq[wi] + smooth / (total_poll + vocabulary_size))
                poll_dict[wi] = [poll_freq[wi], P_wi_poll]
            # not exist
            else:
                P_wi_poll = math.log10(smooth / (total_poll + vocabulary_size))
                # line = line + str(0) + '  ' + str(P_wi_poll) + '  '
                poll_dict[wi] = [0, P_wi_poll]
        # smooth ==0
        else:
            # word exists
            if wi in poll_freq:
                P_wi_poll = math.log10(poll_freq[wi] + smooth / (total_poll + vocabulary_size))
                poll_dict[wi] = [poll_freq[wi], P_wi_poll]
            # not exist
            else:
                P_wi_poll = -float('inf')
                poll_dict[wi] = [0, P_wi_poll]

    if task == 1:
        # Save model in model-2018.txt
        save_model_txt('model-2018.txt', vocabulary, story_dict, ask_hn_dict, show_hn_dict, poll_dict)
    elif task == 2:
        save_model_txt('stopword-model.txt', vocabulary, story_dict, ask_hn_dict, show_hn_dict, poll_dict)
    elif task == 3:
        save_model_txt('wordlength-model.txt', vocabulary, story_dict, ask_hn_dict, show_hn_dict, poll_dict)
    return story_dict, ask_hn_dict, show_hn_dict, poll_dict


# Save model in model-2018.txt
def save_model_txt(filename, vocabulary, story_dict, ask_hn_dict, show_hn_dict, poll_dict):
    file = open(filename, "w", encoding='utf-8')
    i = 1
    for wi in vocabulary:
        line = str(i) + '  ' + wi + '  ' + str(story_dict[wi][0]) + '  ' + '%.10f' % story_dict[wi][1] + '  ' \
               + str(ask_hn_dict[wi][0]) + '  ' + '%.10f' % ask_hn_dict[wi][1] + '  ' + str(show_hn_dict[wi][0]) + '  ' \
               + '%.10f' % show_hn_dict[wi][1] + '  ' + str(poll_dict[wi][0]) \
               + '  ' + str('%.10f' % poll_dict[wi][1]) + '\n'
        file.write(line)
        i = i + 1
    file.close()


# Save vocabulary to vocabulary.txt Save remove_word.txt
def save_txt_file(file_name, content):
    file = open(file_name, "w", encoding='utf-8')
    for word in content:
        file.write(word + '\n')
    file.close()


# Create Vocabulary
def Create_vocabulary(story_list, ask_hn_list, show_hn_list, poll_list):
    vocabulary_list = story_list + ask_hn_list + show_hn_list + poll_list
    vocabulary_Series = pd.Series(vocabulary_list)
    vocabulary_freq = vocabulary_Series.value_counts()

    return vocabulary_freq, list(vocabulary_freq.index.values)


def processing_data_set(data):
    data_numpy = data.to_numpy()
    print(data_numpy)
    # words
    words = []
    i = 1
    for d in data_numpy:
        print("\r Loading... ".format(i) + str(i), end="")
        i = i+1
        # start
        d = ''.join(non_ascii(d))
        d = d.lower()
        words = words + filter_character(d)
    return words


def filter_character(sentence):
    # step1 replace Ask hn, story hn, special character
    special_character = []
    # process Ask HN, Story HN, Poll
    sentence = sentence.replace('Ask HN:', 'ask_hn')
    sentence = sentence.replace('Show HN:', 'show_hn')
    sentence = sentence.replace('Poll:', 'poll')

    # process special characters
    words = sentence.split(' ')
    for word in words:
        if set('[~@#$^&*_-+{}]+$').intersection(word):
            special_character.append(word)
            sentence = sentence.replace(word, '1')
        # brand/business NN
        elif '/' in word:
            new_word = word.split('/')
            sentence = sentence.replace(word, new_word[0]+' '+new_word[1])
    # bi-gram
    bi_list, sentence = bi_gram(sentence)
    special_character = special_character + bi_list
    # one word lemmatization
    lemmatize = lemmatize_all(sentence)
    special_character = special_character + lemmatize
    return special_character


# remove non ascii
def non_ascii(sentence):
    for i in sentence:
        if ord(i) < 128:
            yield i
        else:
            yield ' '


# bi-gram
def bi_gram(sentence):
    bi_list = []
    tokens = nltk.wordpunct_tokenize(sentence)
    bigram = nltk.bigrams(tokens)
    bigram_list = list(bigram)
    for bi in bigram_list:
        bi_word = bi[0] + ' ' + bi[1]
        if bi_word in bi_words.values:
            bi_list.append(bi[0].lower() + ' ' + bi[1].lower())
            sentence = sentence.replace(bi_word, '1')
    return bi_list, sentence


# word lemmatization
def lemmatize_all(sentence):
    word_list = []
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        # noun
        if tag.startswith('NN') and word.isalpha():
            for w, t in pos_tag(word_tokenize(word.lower())):
                if t.startswith('VB'):
                    # Did
                    word_list.append(wnl.lemmatize(w.lower(), pos='v'))
                else:
                    word_list.append(wnl.lemmatize(word.lower(), pos='n'))
        # U.S.A
        elif tag.startswith('NN') and set('.+$').intersection(word):
            word_list.append(word)
        # verb
        elif tag.startswith('VB') and word.isalpha():
            word_list.append(wnl.lemmatize(word.lower(), pos='v'))
        # adj
        elif tag.startswith('JJ') and word.isalpha():
            word_list.append(wnl.lemmatize(word.lower(), pos='a'))
        # adv
        elif tag.startswith('R') and word.isalpha():
            word_list.append(wnl.lemmatize(word.lower(), pos='r'))
        else:
            if task == 1:
                remove_word.append(word)
    return word_list


if __name__ == "__main__":
    main()
