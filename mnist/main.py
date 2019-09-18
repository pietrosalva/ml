
import pickle, gzip
import numpy as np
import matplotlib.pyplot as plt
DATASET_FILENAME = r"D:\Dropbox\ProjectX\ml\data\mnist\mnist.pkl.gz"
import multiclass


def visualizeSet(in_set, title, perLabel = False) :


    # Get the list of possible labels
    label_list = np.unique(train_set[1])

    if perLabel :

        for curr_label\
                in label_list:

            # Visualize the dataset
            c_list = []
            for l, v in zip(in_set[1], in_set[0]):
                if l == k:
                    c_list.append(v)
            num_samples = len(c_list)
            num_cols = int(np.sqrt(num_samples))
            num_rows = int(np.ceil(float(num_samples)/num_cols))
            num_samples_per_page = num_samples
            sample_size = int(np.sqrt(len(in_set[0][0])))
            canvas = np.zeros((sample_size * num_rows, sample_size * num_cols), int)

            for k in range(len(c_list)):
                img = 255 * c_list[k].reshape((sample_size, sample_size))
                img = img.astype(int)
                r = int(np.floor(k / num_cols))
                c = k % num_cols
                canvas[r*sample_size:(r+1)*sample_size, c*sample_size:(c+1)*sample_size] = img
            fig = plt.figure()
            plt.imshow(canvas, extent=[0, 1, 0, 1])
            plt.title(title + " label: " + str(curr_label))
            plt.show()


    else :

        # Visualize the dataset
        num_samples = len(in_set[0])
        num_cols = 20
        num_rows = 20
        num_samples_per_page = num_rows * num_cols
        sample_size = int(np.sqrt(len(in_set[0][0])))
        canvas = np.zeros((sample_size*num_rows, sample_size*num_cols), int)
        for k in range(num_samples_per_page):
            img = 255 * in_set[0][k].reshape((sample_size, sample_size))
            img = img.astype(int)
            r = int(np.floor(k / num_cols))
            c = k % num_cols
            canvas[r*sample_size:(r+1)*sample_size, c*sample_size:(c+1)*sample_size] = img
        fig = plt.figure()
        plt.imshow(canvas, extent=[0, 1, 0, 1])
        plt.title(title)
        plt.show()




if __name__ == "__main__" :

    # Load the dataset
    f = gzip.open(DATASET_FILENAME, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    print("Data loaded")

    visualizeSet(train_set, "Training set", perLabel = true)
    visualizeSet(valid_set, "Validation set", perLabel = true)
    visualizeSet(test_set, "Testing set", perLabel = true)


    # Train the model

    # Test




