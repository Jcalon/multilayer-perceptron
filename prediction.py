from src.data import Datasets
from src.multilayer_perceptron import MultiLayerPerceptron as mlp
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform a prediction on a given set, then evaluate it using the binary cross-entropy error function.')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the dataset.')
    args = parser.parse_args()
    data = args.data
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
        dataset = Datasets(data, "Malin/Benin", features)
        mlp = mlp(
            dataset=dataset,
            filepath="./model.json",
            train_set_percent=0
        )
        print(mlp.predictions(dataset.X_test))
        mlp.get_accuracy()
        print(f"Accuracy: {mlp.accuracy}")
    except Exception as e:
        print(e)
        exit(1)