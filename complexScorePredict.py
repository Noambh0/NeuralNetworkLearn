import numpy as np

study_hours = float(input("Enter study hours: "))
sleep_hours = float(input("Enter sleep hours: "))
previous_score = float(input("Enter previous score: "))

total_weight = np.array([study_hours, sleep_hours, previous_score], dtype=float)

w1 = np.random.randn()
w2 = np.random.randn()
w3 = np.random.randn()
weight = np.array([w1, w2, w3], dtype=float)

bias = np.random.randn()

X = np.array([
    [1, 8, 55],
    [2, 7, 60],
    [3, 6, 65],
    [4, 8, 70],
    [5, 7, 75],
    [6, 6, 78],
    [7, 8, 82],
    [8, 7, 85],
    [9, 6, 88],
    [10, 8, 92]
])

y = np.array([
    58,
    62,
    66,
    72,
    76,
    79,
    84,
    87,
    90,
    95
])

learning_rate = 0.00001

for epoch in range(100000):
    prediction = (X @ weight) + bias
    error = prediction - y

    weight_gradient = np.mean(
        2 * error[:, np.newaxis] * X,
        axis=0
    )

    bias_gradient = np.mean(2 * error)

    weight -= learning_rate * weight_gradient
    bias -= learning_rate * bias_gradient

    if epoch % 100 == 0:
        print(
            "Epoch:", epoch,
            "Weight:", weight,
            "Bias:", bias
        )

print("\nTraining finished")
print("Weight:", weight)
print("Bias:", bias)

predicted_score = float((total_weight @ weight) + bias)
print("you predicted score is: ", round(predicted_score))
