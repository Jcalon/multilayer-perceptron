import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a dataset into train and test sets.')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the dataset.')
    parser.add_argument('--percent_train', type=int, default=80,
                        help='Percent of the dataset to use for training.')
    args = parser.parse_args()
    data = args.data
    percent_train = args.percent_train
    if percent_train > 100 or percent_train < 0:
        raise ValueError("percent_train must be between 0 and 100")
    data = pd.read_csv(data)
    data = data.sample(frac=1).reset_index(drop=True)
    train_samples = int(percent_train/100 * len(data))
    test_data = data.iloc[train_samples:, :]
    train_data = data.iloc[:train_samples, :]
    train_data.to_csv('./data/train_dataset.csv', index=False)
    test_data.to_csv('./data/test_dataset.csv', index=False)