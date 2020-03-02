import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def fitTime():
    labels =["perceptron", "svm_l", "svm_nl", "dt_gini", "dt_entropy", "knn_minkowski", "knn_euclidean", "knn_manhattan"]
    digit_fit = [14.35498046875, 38.466796875, 51.1171875, 14.032958984375, 22.347900390625, 2.47607421875, 2.26708984375, 2.166015625]
    ozone_fit = [1.9267578125, 119.163330078125, 27.617919921875, 15.680908203125, 31.794189453125, 10.79296875, 10.634033203125, 11.482177734375]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, digit_fit, width, label='Digits')
    rects2 = ax.bar(x + width/2, ozone_fit, width, label='Ozone')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time (ms)')
    ax.set_title('Time to Fit')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    fig.tight_layout()

    plt.savefig("fit.png")

def predTime():
    labels =["perceptron", "svm_l", "svm_nl", "dt_gini", "dt_entropy", "knn_minkowski", "knn_euclidean", "knn_manhattan"]
    digit_pred = [0.4658203125, 13.931884765625, 28.14697265625, 0.27099609375, 0.18212890625, 81.09814453125, 73.896728515625, 69.878173828125]
    ozone_pred = [0.326171875, 10.845947265625, 12.5009765625, 3.100830078125, 3.159912109375, 72.156005859375, 40.14306640625, 69.466796875]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, digit_pred, width, label='Digits')
    rects2 = ax.bar(x + width/2, ozone_pred, width, label='Ozone')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time (ms)')
    ax.set_title('Time to pred')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    fig.tight_layout()

    plt.show()

def trainAccuracy():
    labels =["perceptron", "svm_l", "svm_nl", "dt_gini", "dt_entropy", "knn_minkowski", "knn_euclidean", "knn_manhattan"]
    digit = [0.9602, 1, 0.9928, 0.9832, 0.9809, 0.9912, 1, 1]
    ozone = [0.9055, 0.9358, 0.9365, 0.9705, 0.9280, 0.9388, 0.9373, 0.9389]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, digit, width, label='Digits')
    rects2 = ax.bar(x + width/2, ozone, width, label='Ozone')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy Score')
    ax.set_title('Training Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    fig.tight_layout()

    plt.show()

def testAccuracy():
    labels =["perceptron", "svm_l", "svm_nl", "dt_gini", "dt_entropy", "knn_minkowski", "knn_euclidean", "knn_manhattan"]
    digit = [0.9296, 0.9759, 0.9833, 0.8389, 0.8574, 0.9926, 0.9907, 0.9833]
    ozone = [0.8953, 0.9242, 0.9170, 0.9188, 0.9422, 0.9332, 0.9332, 0.9368]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, digit, width, label='Digits')
    rects2 = ax.bar(x + width/2, ozone, width, label='Ozone')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy Score')
    ax.set_title('Testing Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    fig.tight_layout()

    plt.show()

def accuracyOverTime():
    labels =["perceptron", "svm_l", "svm_nl", "dt_gini", "dt_entropy", "knn_minkowski", "knn_euclidean", "knn_manhattan"]
    digit = [0.9296/25.41, 0.9759/76.55, 0.9833/136.78, 0.8389/14.972, 0.8574/15.75, 0.9926/251.37, 0.9907/114.02, 0.9833/105.89]
    ozone = [0.8953/5.29, 0.9242/138.72, 0.9170/62.065, 0.9188/48.41, 0.9422/26.98, 0.9332/231.01, 0.9332/132.41, 0.9368/223.71]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, digit, width, label='Digits')
    rects2 = ax.bar(x + width/2, ozone, width, label='Ozone')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy Score / Time')
    ax.set_title('Testing Accuracy / Runtime')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    fig.tight_layout()

    plt.show()

fitTime()
predTime()
trainAccuracy()
testAccuracy()
accuracyOverTime()