import gzip
import json
import pickle
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional


data = [[]]
labels = [[]]
for fold in range(1,11):
    print("Loading fold", fold)
    pred_file = open("../output/pred_fold-%d.txt" % fold)
    #pred_file = open("output/pred.txt")
    data_file = gzip.open("../data/train-10-fold-prep/fulldoc-fold-%d-test-processed.jsonl.gz" % fold)
    _ = data_file.readline()
    last_doc = None
    for pred_line, data_line in zip(pred_file, data_file):
        probs = json.loads('['+pred_line.replace('\"','')+']')[0]
        datum = json.loads(data_line)
        labs = datum[1]
        doc = datum[2]
        if doc != last_doc:
            if last_doc is not None:
                # New doc
                data.append([])
                labels.append([])
            last_doc = doc
        data[-1].append(probs)
        labels[-1] = labs


# Save data for possible future use
print("Saving...")
#json.dump({'probs': data, 'labels': labels}, gzip.open("probs.json.gz", 'wt', encoding="ascii"))
pickle.dump({'probs': data, 'labels': labels}, open("probs.pickle", 'wb'))

print("Preparing...")
seq_len = max([len(d) for d in data])
feat_dim = len(data[0][0])
padded_data = [[[-1.]*feat_dim]*(seq_len-len(d))+d for d in data]
array = np.array(padded_data)
#np.save("probs.npy", array) # Padded array is huge

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)


input = Input(shape=(array.shape[1],array.shape[2]))
lstm = Bidirectional(LSTM(array.shape[2]))(input)
output = Dense(array.shape[2], activation='sigmoid')(lstm)
model = Model(inputs=input, outputs=output)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

hist = model.fit(array, labels, batch_size=32, verbose=1, epochs=50, validation_split=0.1)
