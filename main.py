from model import DataExtractor, DataCleaning, MachineLearningAlgorithm

if __name__ == '__main__':
    extractor = DataExtractor(n_pages = 43)
    extractor.extracting_houses_links()
    extractor.extracting_features()

    cleaner = DataCleaning(extractor)
    cleaner.evaluating_frequent_features()
    cleaner.cleaning_data()
    cleaner.structing_data()

    algorithm = MachineLearningAlgorithm(cleaner)
    algorithm.preparing_data()
    algorithm.training()
    algorithm.testing()
    #algorithm.predict(some_imput)

