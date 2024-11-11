def fractional_knapsack(value, weight, capacity):
    ratio = [v / w for v, w in zip(value, weight)]  
    index = sorted(range(len(value)), key=lambda i: ratio[i], reverse=True)

    max_value = 0
    fractions = [0] * len(value)

    for i in index:
        if weight[i] <= capacity:
            fractions[i] = 1  
            max_value += value[i]
            capacity -= weight[i]
        else:
            fractions[i] = capacity / weight[i] 
            max_value += value[i] * (capacity / weight[i])
            break  

    return max_value, fractions


if __name__ == "__main__":
    # Input the number of items
    n = int(input("Enter the number of items: "))

    value = []  # List to store values of the items
    weight = []  # List to store weights of the items

    # Collect value and weight for each item using a for loop
    for i in range(n):
        v = int(input(f"Enter the value of item {i+1}: "))
        w = int(input(f"Enter the weight of item {i+1}: "))
        value.append(v)
        weight.append(w)

    # Input the capacity of the knapsack
    capacity = int(input("Enter the maximum weight capacity of the knapsack: "))

    # Call the fractional knapsack function to get the result
    max_value, fractions = fractional_knapsack(value, weight, capacity)

    # Output the results
    print(f"\nThe maximum value that can be carried: {max_value}")
    print(f"The fractions of items to be taken: {fractions}")
