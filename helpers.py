import os
import csv
import numpy as np

def load_csv_data(data_path=os.path.join("data")):

    # print("files:", os.listdir(DATA_DIR))

    y_raw = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
    )

    ids_y = y_raw[:, 0].astype(np.int64)
    y_train = y_raw[:,1].astype(int)

    x_raw = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"),
        delimiter=",",
        skip_header=1,
        filling_values=np.nan,
    )

    ids_x = x_raw[:, 0].astype(np.int64)
    x_train = x_raw[:, 1:]

    if not np.array_equal(ids_x, ids_y):
        raise ValueError("Ids in x_train.csv and y_train.csv do not match.")

    x_test_raw = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"),
        delimiter=",",
        skip_header=1,
        filling_values=np.nan,
    )

    ids_test = x_test_raw[:, 0].astype(np.int64)
    x_test = x_test_raw[:, 1:]

    return y_train, x_train, ids_x, x_test, ids_test


def create_csv_submission(ids, y_pred, name):
    with open(name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id", "Prediction"])
        for i, pred in zip(ids, y_pred):
            writer.writerow([int(i), int(pred)])
