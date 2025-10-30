from typing import Dict, Sized, Union

import logging
import matplotlib.pyplot as plt


def plot_pie_chart_with_counts(
    X_train: Sized, X_test: Sized, X_test_orig: Sized
) -> None:
    """Plots a pie chart showing the relative sizes of datasets.

    Args:
        X_train (Sized): The training dataset.
        X_test (Sized): The test dataset.
        X_test_orig (Sized): The original, unsplit test dataset.
    """
    sizes = [len(X_train), len(X_test), len(X_test_orig)]
    labels = ["X_train", "X_test", "X_test_orig"]

    if sum(sizes) == 0:
        logger = logging.getLogger("ml_grid")
        logger.warning("Cannot plot pie chart, all datasets are empty.")
        return

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


def plot_dict_values(data_dict: Dict[str, bool]) -> None:
    """Creates a horizontal bar chart representing boolean values in a dictionary.

    Args:
        data_dict (Dict[str, bool]): A dictionary with string keys and boolean
            values.
    """
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


def create_bar_chart(
    data_dict: Dict[str, Union[int, float]],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> None:
    """Creates a horizontal bar chart from a dictionary of data.

    Args:
        data_dict (Dict[str, Union[int, float]]): Dictionary with category names
            as keys and their corresponding values.
        title (str, optional): The title of the chart. Defaults to "".
        x_label (str, optional): The label for the x-axis. Defaults to "".
        y_label (str, optional): The label for the y-axis. Defaults to "".
    """

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


def plot_candidate_feature_category_lists(data: Dict[str, int]) -> None:
    """Plots a bar chart for candidate feature category counts.

    Args:
        data (Dict[str, int]): A dictionary where keys are feature category
            names and values are the counts of features in that category.
    """
    create_bar_chart(
        data, title="Feature category counts", x_label="features", y_label="counts"
    )
