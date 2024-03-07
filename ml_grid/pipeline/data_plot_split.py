import matplotlib.pyplot as plt

# class plot_methods():

# def __init__(self):


def plot_pie_chart_with_counts(X_train, X_test, X_test_orig):
    """
    Plot a pie chart with value counts for X_train, X_test, and X_test_orig.

    Parameters:
        X_train (list or array): The training dataset.
        X_test (list or array): The test dataset.
        X_test_orig (list or array): The original test dataset.

    Returns:
        None
    """
    sizes = [len(X_train), len(X_test), len(X_test_orig)]
    labels = ["X_train", "X_test", "X_test_orig"]

    # Colors for the pie chart sections
    colors = ["#ff9999", "#66b3ff", "#c2c2f0"]

    # Explode the section with the largest size
    explode = tuple(
        0.1 if i == sizes.index(max(sizes)) else 0 for i in range(len(sizes))
    )

    # Create the pie chart
    plt.figure(figsize=(2, 2))
    patches, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        explode=explode,
    )

    # Equal aspect ratio ensures that the pie chart is drawn as a circle.
    plt.axis("equal")

    # Add a title
    plt.title("Sizes of Datasets")

    # Add value counts to the plot
    for i, text in enumerate(texts):
        percent = sizes[i] / sum(sizes) * 100
        text.set_text(f"{labels[i]}: {sizes[i]} ({percent:.1f}%)")

    plt.show()


def plot_dict_values(data_dict):
    # Extract the keys and values from the dictionary
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    # Define colors for True and False values
    colors = ["green" if val else "red" for val in values]

    # Create the horizontal bar chart
    plt.figure(figsize=(2, 2))
    plt.barh(keys, [1] * len(keys), color=colors)
    plt.yticks(rotation=0)  # Keep y-axis ticks horizontal

    # Add a legend for the colors
    legend_colors = [
        plt.Rectangle((0, 0), 1, 1, fc="green", edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor="none"),
    ]
    plt.legend(legend_colors, ["True", "False"], loc="upper right")

    # Set axis labels and title
    plt.xlabel("Value")
    plt.ylabel("Data Fields")
    plt.title("Values for the Given Dictionary")

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Given dictionary


# def create_bar_chart(data_dict, title='', x_label='', y_label=''):
#     # Extracting keys and values from the dictionary
#     categories = list(data_dict.keys())
#     values = list(data_dict.values())

#     # Setting up the bar chart
#     plt.bar(categories, values)

#     # Adding labels and title
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)

#     # Displaying the chart
#     plt.show()


def create_bar_chart(data_dict, title="", x_label="", y_label=""):
    # Extracting keys and values from the dictionary
    categories = list(data_dict.keys())
    values = list(data_dict.values())

    # Setting up the horizontal bar chart
    plt.barh(categories, values)

    # Adding labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Adding angled labels
    for index, value in enumerate(values):
        plt.text(
            value,
            index,
            str(value),
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    # Displaying the chart
    plt.show()


def plot_candidate_feature_category_lists(data):

    create_bar_chart(
        data, title="Feature category counts", x_label="features", y_label="counts"
    )
