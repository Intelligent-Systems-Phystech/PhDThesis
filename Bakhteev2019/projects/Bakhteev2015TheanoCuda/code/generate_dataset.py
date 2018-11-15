fname = 'WISDM_ar_v1.1_raw.txt'
import numpy as np
import csv, sys


def preprocess_csv():
    classes = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']

    i = 0
    with open(fname) as inp:
        with open(fname + '_preprocessed', 'w') as out:
            for line in inp:
                i += 1
                splits = line.split(',')
                if len(splits) < 3:
                    print ('error, line:', i)
                to_write = splits[0] + ',' + str(classes.index(splits[1])) + ',' + ','.join(splits[2:])[:-2]
                if to_write[-1] == ',':
                    to_write = to_write[:-1]
                out.write(to_write + '\n')


def csv_to_batches():
    data = np.genfromtxt(fname + '_preprocessed', dtype=float, delimiter=',', invalid_raise=False)
    result = []
    # print data.shape
    i = 0
    while i < data.shape[0] - 200:
        if np.all(data[i:i + 200, 1] != data[i + 200, 1]):
            i += 1
            continue

        line = np.array(np.concatenate((np.array([data[i, 1]]), data[i:i + 200, 3:].flatten()), axis=1))
        result.append(line)
        i += 200
        print i
    result_np = np.array(result)
    print result_np.shape
    np.save('WISDM', result_np)


preprocess_csv()
data = csv_to_batches()
"""
Data can be simpily balanced by next code:
# balancing data
    for c in range(0, 6):
        datas.append(np.repeat(np.where(data[:, 0] == c), 20))
    for c in range(1, 6):
        delta = sum(data[:, 0] == 0) - sum(data[:, 0] == c)
        if delta > 0:
            ready = False
        np.random.shuffle(datas[c])
        data = np.concatenate((data, data[datas[c][:delta], :]))
    data_mean = np.mean(data[:, 1:], axis=0)
    data_std = np.std(data[:, 1:], axis=0)
    # standartize
    data[:, 1:] = (data[:, 1:] - data_mean) / data_std
    np.random.shuffle(data)
"""
