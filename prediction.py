from src.data import Datasets
from src.multilayer_perceptron import MultiLayerPerceptron as mlp

if __name__ == '__main__':
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
    dataset = Datasets('./data/data.csv', "Malin/Benin", features)
    mlp = mlp(
        dataset=dataset,
        filepath="./model.json"
    )
    print(mlp.get_accuracy())
    print(dataset.Y)
    print(mlp.predictions(dataset.X_train))