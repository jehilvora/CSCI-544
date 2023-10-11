



import json
import numpy as np
from collections import Counter, defaultdict






TRAIN_DATA_PATH = 'data/train.json'
DEV_DATA_PATH = 'data/dev.json'
TEST_DATA_PATH = 'data/test.json'
UNKNOWN_KEY = '<unk>'
THRESHOLD = 2
OUTPUT_FOLDER = 'verification/out'
OUTPUT_PATH_VOCAB = OUTPUT_FOLDER + '/vocab.txt'
OUTPUT_PATH_HMM = OUTPUT_FOLDER + '/hmm.json'
OUTPUT_PATH_GREEDY = OUTPUT_FOLDER + '/greedy.json'
OUTPUT_PATH_VITERBI = OUTPUT_FOLDER + '/viterbi.json'





initial_counts = {}
transition_counts = defaultdict(lambda: defaultdict(int))
emission_counts = defaultdict(lambda: defaultdict(int))
with open(TRAIN_DATA_PATH) as f:
    train_data = json.load(f)
    train_data_words = []
    for train_entry in train_data:
        train_data_words.extend(train_entry['sentence'])

temp_dict = Counter(train_data_words)

freq_dict = {}

freq_dict[UNKNOWN_KEY] = 0
for word in temp_dict:
    if temp_dict[word] < THRESHOLD:
        freq_dict[UNKNOWN_KEY] += temp_dict[word]
    else:
        freq_dict[word] = temp_dict[word]

unk_value = freq_dict[UNKNOWN_KEY]
del freq_dict[UNKNOWN_KEY]
freq_dict = dict([(UNKNOWN_KEY, unk_value)] + sorted(freq_dict.items(), key=lambda item: item[1], reverse=True))

with open(OUTPUT_PATH_VOCAB, 'w') as f:
    for o, word in enumerate(freq_dict):
        freq_dict[word] = {
            'index': o,
            'frequency': freq_dict[word]
        }
        f.write(f'{word}\t{o}\t{freq_dict[word]["frequency"]}\n')


print(f"What threshold value did you choose for identifying unknown words for replacement? : {THRESHOLD}")
print(f"What is the overall size of your vocabulary? : {len(freq_dict)}")
print(f"How many times does the special token '< unk >' occur following the replacement process? : {freq_dict['<unk>']['frequency']}")























tags = {}
for train_entry in train_data:
    labels = train_entry['labels']
    label_len = len(labels)
    for s in range(label_len):
        tag = labels[s]
        if tag not in tags:
            tags[tag] = {
                'index': len(tags),
                'frequency': 1
            }
        else:
            tags[tag]['frequency'] += 1
        if s == 0:
            initial_counts[tag] = initial_counts.get(tag, 0) + 1
        emitted_word = train_entry['sentence'][s] if train_entry['sentence'][s] in freq_dict else UNKNOWN_KEY
        emission_counts[tag][emitted_word] += 1
        if s < label_len - 1:
            next_tag = labels[s + 1]
            transition_counts[tag][next_tag] += 1
            tags[tag]['trans_freq'] = tags[tag].get('trans_freq', 0) + 1
            

NUM_TAGS = len(tags)
NUM_WORDS = len(freq_dict)
tag_list = list(tags.keys())


transition = {}
emission = {}

for tag in transition_counts:
    for next_tag in transition_counts[tag]:
        transition[f"('{tag}', '{next_tag}')"] = transition_counts[tag][next_tag] / tags[tag]['trans_freq']

for tag in emission_counts:
    for word in emission_counts[tag]:
        emission[f"('{tag}', '{word}')"] = emission_counts[tag][word] / tags[tag]['frequency']

hmm_json = {
    'transition': transition,
    'emission': emission,
}

with open(OUTPUT_PATH_HMM, 'w') as json_file:
    json.dump(hmm_json, json_file)


print(f"Transition parameters:  {len(transition)}")
print(f"Emission parameters:  {len(emission)}")














initial_prob = np.zeros(NUM_TAGS)
for o, tag in enumerate(tags):
    initial_prob[o] = initial_counts.get(tag, 0) / len(train_data)

transition_prob = np.zeros((NUM_TAGS, NUM_TAGS))
for tag in transition_counts:
    for next_tag in transition_counts[tag]:
        transition_prob[tags[tag]['index']][tags[next_tag]['index']] = transition_counts[tag][next_tag] / tags[tag]['trans_freq']

emission_prob = np.zeros((NUM_TAGS, NUM_WORDS))
for tag in emission_counts:
    for word in emission_counts[tag]:
        emission_prob[tags[tag]['index']][freq_dict[word]['index']] = emission_counts[tag][word] / tags[tag]['frequency']





with open(DEV_DATA_PATH) as f:
    dev_data = json.load(f)

greedy = []
res = np.array([], dtype=bool)

for data_idx, test_entry in enumerate(dev_data):
    sentence = test_entry['sentence']
    pred = []
    for o, word in enumerate(sentence):
        init_prob = initial_prob if o == 0 else transition_prob[tags[pred[-1]]['index']]
        mul_value = init_prob * emission_prob[:, freq_dict.get(word, freq_dict[UNKNOWN_KEY])['index']]
        pred.append(tag_list[np.argmax(mul_value)])
    greedy.append({
            'index': data_idx,
            'sentence': sentence,
            'labels': pred
        })
    res = np.append(res, np.array(pred) == np.array(test_entry['labels']))
print(f"Accuracy of greedy algorithm on dev data: {res.mean()}")








with open(TEST_DATA_PATH) as f:
    test_data = json.load(f)
greedy = []

for data_idx, test_entry in enumerate(test_data):
    sentence = test_entry['sentence']
    pred = []
    for o, word in enumerate(sentence):
        init_prob = initial_prob if o == 0 else transition_prob[tags[pred[-1]]['index']]
        mul_value = init_prob * emission_prob[:, freq_dict.get(word, freq_dict[UNKNOWN_KEY])['index']]
        pred.append(tag_list[np.argmax(mul_value)])
    greedy.append({
            'index': data_idx,
            'sentence': sentence,
            'labels': pred
        })
with open(OUTPUT_PATH_GREEDY, 'w') as json_file:
    json.dump(greedy, json_file)





def viterbi(sentence, initial_prob, transition_prob, emission_prob):

    text_length = len(sentence)
    viterbi_matrix = np.zeros((NUM_TAGS, text_length))
    backpointers = np.zeros((NUM_TAGS, text_length), dtype=int)

    word = sentence[0] if sentence[0] in freq_dict else UNKNOWN_KEY
    word_index = freq_dict[word]['index']
    viterbi_matrix[:, 0] = initial_prob * emission_prob[:, word_index]

    for t in range(1, len(sentence)):
        for i, tag in enumerate(tags):
            word = sentence[t] if sentence[t] in freq_dict else UNKNOWN_KEY
            word_index = freq_dict[word]['index']

            max_prob = 0
            max_index = 0
            for j, prev_tag in enumerate(tags):
                transition = transition_prob[j, i]
                emission = emission_prob[i, word_index]
                prob = viterbi_matrix[j, t - 1] * transition * emission

                if prob > max_prob:
                    max_prob = prob
                    max_index = j

            viterbi_matrix[i, t] = max_prob
            backpointers[i, t] = max_index

    
    best_sequence = []
    max_prob = 0
    max_index = 0
    for i in range(NUM_TAGS):
        if viterbi_matrix[i, len(sentence) - 1] > max_prob:
            max_prob = viterbi_matrix[i, len(sentence) - 1]
            max_index = i

    best_sequence.append(max_index)
    for t in range(len(sentence) - 1, 0, -1):
        max_index = backpointers[max_index, t]
        best_sequence.append(max_index)

    best_sequence = best_sequence[::-1]  

    return [tag_list[i] for i in best_sequence]





predictions = []
true_labels = []
viterbi_res = []
for data_idx, test_entry in enumerate(dev_data):
    sentence = test_entry['sentence']
    true_label = test_entry['labels']
    predicted_label = viterbi(sentence, initial_prob, transition_prob, emission_prob)
    viterbi_res.append({
            'index': data_idx,
            'sentence': sentence,
            'labels': predicted_label
        })
    predictions.extend(predicted_label)
    true_labels.extend(true_label)

accuracy = (np.array(predictions) == np.array(true_labels)).mean()
print("Accuracy of viterbi algorithm on dev data:", accuracy)









with open(TEST_DATA_PATH) as f:
    test_data = json.load(f)
viterbi_res = []
for data_idx, test_entry in enumerate(test_data):
    sentence = test_entry['sentence']
    predicted_label = viterbi(sentence, initial_prob, transition_prob, emission_prob)
    viterbi_res.append({
            'index': data_idx,
            'sentence': sentence,
            'labels': predicted_label
        })

with open(OUTPUT_PATH_VITERBI, 'w') as json_file:
    json.dump(viterbi_res, json_file)


