from src.models.evaluate import evaluate_on_external_data


def main():
    data_path = "data/raw/test_data.csv"

    evaluate_on_external_data(data_path)


if __name__ == "__main__":
    main()