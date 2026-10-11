import numpy as np

hour_study = float(input("how many hours have you studied? "))

weight = np.random.randn()
bias = np.random.randn()

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

y = np.array([8, 17, 26, 34, 43, 53, 61, 72, 78, 89, 96], dtype=float)

learning_rate = 0.001

for epoch in range(10000):
    prediction = x * weight + bias

    loss = np.mean((prediction - y) ** 2)

    error = prediction - y

    weight_gradient = np.mean(2 * error * x)
    bias_gradient = np.mean(2 * error)

    weight -= learning_rate * weight_gradient
    bias -= learning_rate * bias_gradient

    if epoch % 100 == 0:
        print(
            "Epoch:", epoch,
            "Loss:", loss,
            "Weight:", weight,
            "Bias:", bias
        )

rounded_weight = np.round(weight)
rounded_bias = np.round(bias)

new_y = []

for i in x:
    index_x = i*rounded_weight + rounded_bias
    new_y.append(index_x)

print("\nTraining finished")
print("Weight:", weight)
print("Bias:", bias)
print("y = x" ,rounded_weight, " " ,rounded_bias)

print(new_y == y)

print(hour_study * rounded_weight + rounded_bias)