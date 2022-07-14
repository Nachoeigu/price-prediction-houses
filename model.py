import requests
from lxml import html
import time
import re
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataExtractor:

    def __init__(self, n_pages): #See the number of pages your search result has
        self.n_pages = n_pages
        self.enlaces = []
        self.dictionary = {}

    def extracting_houses_links(self): # In this case is about houses in Banfield
        for number in range(1, self.n_pages + 1):
            time.sleep(2.5)
            print(f"Extracting links from the page number {number}")
            headers = {
                'authority': 'www.argenprop.com',
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'accept-language': 'en-US;q=0.7',
                'cache-control': 'max-age=0',
                'referer': 'https://www.argenprop.com/casa-venta-partido-lomas-de-zamora',
                'sec-fetch-dest': 'document',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-site': 'same-origin',
                'sec-fetch-user': '?1',
                'sec-gpc': '1',
                'upgrade-insecure-requests': '1',
                'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.61 Safari/537.36',
            }

            response = requests.get(f'https://www.argenprop.com/casa-venta-localidad-banfield-pagina-{number}', headers=headers)

            response = response.content.decode("utf-8")

            data = html.fromstring(response)

            links = data.xpath('//div[@class="listing__item "]//a[contains(@href,"venta")]/@href')
            
            [self.enlaces.append(element) for element in links]

    def extracting_features(self):
        for idx, enlace in enumerate(self.enlaces):
            time.sleep(3)
            print(f"Extracting features from house number {idx + 1}")
            headers = {
                'authority': 'www.argenprop.com',
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'accept-language': 'en-US;q=0.7',
                'cache-control': 'max-age=0',
                'referer': 'https://www.argenprop.com/casa-venta-partido-lomas-de-zamora',
                'sec-fetch-dest': 'document',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-site': 'same-origin',
                'sec-fetch-user': '?1',
                'sec-gpc': '1',
                'upgrade-insecure-requests': '1',
                'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.61 Safari/537.36',
            }

            response = requests.get(f'https://www.argenprop.com{enlace}', headers=headers)
            
            response = response.content.decode("utf-8")
            
            data = html.fromstring(response)

            self.dictionary[str(idx)] = {}


            self.dictionary[str(idx)]['enlace'] = enlace
            feature_items_in_bold = data.xpath("//section//div[@class='property-features-title']/parent::section//ul//li//p/parent::li")
            for feature in feature_items_in_bold:
                feature_name = feature.xpath('.//p')[0].text.strip().lower().replace(':','')
                feature_value = feature.xpath('.//strong')[0].text.replace('m2','').strip().lower()
                self.dictionary[str(idx)][feature_name] = feature_value

            address = data.xpath("//*[@class='titlebar__address']")[0].text.strip().replace('al','').lower()
            address = re.sub("[0-9]+","", address).strip().replace('  ',' ')
            price = data.xpath("//p[@class='titlebar__price']")[0].text.replace('USD','').strip().replace('.','')

            self.dictionary[str(idx)]['precio'] = price
            self.dictionary[str(idx)]['calle'] = address

class DataCleaning(DataExtractor):

    def __init__(self, data_extractor):
        self.dictionary = data_extractor.dictionary
        self.enlaces = data_extractor.enlaces
    
    def evaluating_frequent_features(self):
        all_features = []
        for key in self.dictionary:
            for element in self.dictionary[key]:
                all_features.append(element)
        
        #In this line we see in how many cases we have each feature
        print(pd.DataFrame({'all_features':all_features})['all_features'].value_counts())

        #By default we will consider only the followings
        self.features = ['precio', 'enlace', 'calle', 'cant. dormitorios', 'cant. baños','cant. toilettes','sup. cubierta', 'sup. terreno', 'cant. cocheras', 'antiguedad', 'cant. plantas']

    def cleaning_data(self):
        self.cleaned_dict = {}

        for idx, _ in enumerate(self.enlaces):
            variable = 0
            adaption = {}
            x = self.dictionary[str(idx)]
            
            for feature in self.features:
                try:
                    if feature == 'cant. toilettes':
                        adaption['cant. baños'] = x[feature]
                    else:
                        adaption[feature] = x[feature]
                    
                except:
                    if feature == 'cant. cocheras':
                        adaption['cant. cocheras'] = '0'
                    elif feature == 'cant. plantas':
                        adaption['cant. plantas'] = '1'
                    elif feature == 'cant. toilettes':
                        continue
                    else:
                        variable = 1
                        break
                        
            if variable == 1: #If the case doesnt contain those features, ignore it
                continue
            
            else:
                key_names = list(x.keys())
                for keys in key_names:
                    if keys not in self.features:
                        del x[keys]
                        
                self.cleaned_dict[str(idx)] = adaption

    def structing_data(self):
        self.df = pd.DataFrame()
        for key in self.cleaned_dict:
            self.df = self.df.append(self.cleaned_dict[key], ignore_index = True)

    def export_df(self, file_name):
        self.df.to_csv(f"{file_name}.csv")

    
class MachineLearningAlgorithm(DataCleaning):

    def __init__(self, data_cleaner):
        self.df = data_cleaner.df

    def __cleaning_antiguedad(data):
        if data > 1000:
            return 2022 - data
        else:
            return data

    def preparing_data(self):
        #We clean the data a little more for the algorithm
        self.df = self.df[self.df['precio'] != 'Consultar precio']

        del self.df['calle'] # It is not useful because there are some errors in those strings
        del self.df['enlace']

        self.df['sup. terreno'] = self.df['sup. terreno'].apply(lambda x: x.replace(',','.'))
        self.df = self.df.astype(float)

        self.df['antiguedad'] = self.df['antiguedad'].apply(self.__cleaning_antiguedad) #Some cases we have it as "2004", "1990"

        self.features = self.df.drop(columns = 'precio')
        self.target = self.df[['precio']]
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)
    
    def training(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.target, test_size = 0.2, random_state =123)
        self.model = Ridge()
        self.model.fit(self.X_train, self.y_train)

    def testing(self):
        print(f"Score is: {self.model.score(self.X_test, self.y_test)}")

    def predict(self, input):
        return self.model.predict(input)
