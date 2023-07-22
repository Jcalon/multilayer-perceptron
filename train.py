from src.data import Datasets
from src.multilayer_perceptron import MultiLayerPerceptron as mlp
import argparse


def main(parser):
    args = parser.parse_args()
    layer_sizes = args.layer
    epochs = args.epochs
    loss = args.loss
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    train_set_percent = args.train_set_percent
    data = args.data
    dataset = Datasets(data, "Malin/Benin", features)
    if layer_sizes is None:
        layer_sizes = [24, 24, 24]
    else:
        layer_sizes = [int(layer_size) for layer_size in layer_sizes]
    model = mlp(
        dataset=dataset,
        structure=layer_sizes,
        loss=loss,
        train_set_percent=train_set_percent
    )
    model.train(
        iterations=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    model.save(dataset)
    print('End accuracy :', model.accuracy, '%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a multilayer perceptron.')

    # Add arguments
    parser.add_argument('--layer', nargs='+', type=int, required=False,
                        help='Number of neurons in each layer separated by spaces.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs.')
    parser.add_argument('--loss', choices=['binaryCrossentropy', 'MeanSquaredError'], required=False,
                        help='Loss function for training.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--train_set_percent', type=int, default=80,
                        help='Learning rate for training.')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the dataset.')
    features = [
        "Feature1",
        "Feature2",
        "Feature3",
        "Feature4",
        "Feature5",
        "Feature6",
        "Feature7",
        "Feature8",
        "Feature9",
        "Feature10",
        "Feature11",
        "Feature12",
        "Feature13",
        "Feature14",
        "Feature15",
        "Feature16",
        "Feature17",
        "Feature18",
        "Feature19",
        "Feature20",
        "Feature21",
        "Feature22",
        "Feature23",
        "Feature24",
        "Feature25",
        "Feature26",
        "Feature27",
        "Feature28",
        "Feature29",
        "Feature30"
    ]
    try:
        main(parser)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    except Exception as e:
        print(e)
