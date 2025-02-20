from src.data.preprocessing import DatasetPreprocessor


def main():
    # process dataset
    preprocessor = DatasetPreprocessor()
    preprocessor.extract_dataset()
    preprocessor.resize_dataset(dataframe, (224, 224))

    # train model
    # test model
    # xai
    pass


if __name__ == "__main__":
    main()
