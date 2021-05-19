import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split

def predictionAndCalculateAccurancy(data):
    y = data['label']
    X = data.drop(['label'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    with open('./models/' + 'crop', 'wb') as file:
        pickle.dump(model, file)
    y_pred = model.predict(X_test)
    print("Random Forest Classifier accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)


def getCropsHavingMinimumProbability(crop_features, minimum_Probability):
    with open('./models/' + 'crop', 'rb') as f:
        model = pickle.load(f)
    probabilities = model.predict_proba([crop_features])[0]
    classes = model.classes_
    considered_classes = []
    for ind, probability in enumerate(probabilities):
        if minimum_Probability(probability):
            considered_classes.append(classes[ind])
    return considered_classes

def main():
    crop_data = pd.read_csv("./crop_prediction_dataset.csv")
    predictionAndCalculateAccurancy(crop_data)
if __name__=='__main__':
    main()









