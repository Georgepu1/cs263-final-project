# x = get_sst_data()
def get_sst_data():
    # !pip install datasets
    from datasets import load_dataset
    dataset = load_dataset("sst", "default")
    # Train: 8544 rows, Val: 1101 rows, Test: 2210 rows
    return dataset['train'], dataset['validation'], dataset['test']


# We have split the data into an in-domain set comprised sentences from 17 sources
# and an out-of-domain set comprised of the remaining 6 sources. The in-domain set is
# split into train/dev/test sets, and the out-of-domain is split into dev/test sets.
# The test sets are not made public.
#
# Note: path is to the raw folder
# x = get_cola_data('./data/cola_public/raw/')
def get_cola_data(path):
    train_dev_set = []
    test_set = []
    with open(path + 'in_domain_train.tsv', 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip().split('\t')
            if len(line) == 4:
                # (text, label)
                train_dev_set.append((line[-1], line[1]))
    with open(path + 'in_domain_dev.tsv', 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip().split('\t')
            if len(line) == 4:
                # (text, label)
                test_set.append((line[-1], line[1]))
    train_dev_split = int(len(train_dev_set) * .9)
    return train_dev_set[:train_dev_split], train_dev_set[train_dev_split:], test_set
