"""Linear regression algorithm for machine learning.

For the purpose of this algorithm, i used a csv repo that i got online.



This algorithm will attempt to predict the final grade (G3) that multiple learners will get
by using multiple attributes, such as their first grade (G1), second grade (G2), travel time
(traveltime), study time (studytime), failures (failures) and absences (absences).
"""
from matplotlib import style, pyplot
from numpy import array
from sklearn import *

import pandas as pd


def main():
    __student_data = pd.read_csv("./student-mat.csv", sep=";")  # student-mat-csv included in package

    data = __student_data[["G1", "G2", "G3", "traveltime", "studytime", "failures", "absences"]]

    predict = "G3"

    x = array(data.drop([predict], 1))
    y = array(data[predict])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    # y = mx + b

    print("Coefficient; \n", linear.coef_)
    print("Intercept: \n", linear.intercept_)
    print("\n")
    predictions = linear.predict(x_test)

    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x])

    print(f"Accuracy: {acc}\n")

    for p in data:
        y = "G3"
        style.use("ggplot")
        pyplot.scatter(data[p], data[y])
        pyplot.xlabel(p)
        pyplot.ylabel("Final Grade")
        pyplot.show()


if __name__ == '__main__':
    main()
