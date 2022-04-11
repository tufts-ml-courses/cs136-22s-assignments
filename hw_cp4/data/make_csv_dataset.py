'''

Prereqs
-------
export FASHION_MNIST_REPO_DIR=/path/to/fashion-mnist/
'''

import pandas as pd
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import scipy.interpolate

try:
    assert os.path.exists(os.environ['FASHION_MNIST_REPO_DIR'])
except:
    raise ValueError("Need to set env var FASHION_MNIST_REPO_DIR")

FASHION_MNIST_REPO_DIR = os.environ['FASHION_MNIST_REPO_DIR']
sys.path.append(os.path.join(FASHION_MNIST_REPO_DIR, 'utils'))
import mnist_reader

def show_images(X, label_name, vmin=-1.0, vmax=1.0):
    D = X.shape[1]
    P = int(np.sqrt(D))
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9,9))
    for ii in range(9):
        cur_ax = axes.flatten()[ii]
        cur_ax.imshow(X[ii].reshape(P,P), interpolation='nearest', vmin=vmin, vmax=vmax, cmap='gray')
        cur_ax.set_xticks([])
        cur_ax.set_yticks([])
        cur_ax.set_title(label_name)

def interpolate_2d_img(img_PP, new_shape=(16, 16)):
    W, H = img_PP.shape
    xrange = lambda x: np.linspace(0, 1, x)
    my_interp_func = scipy.interpolate.interp2d(xrange(W), xrange(H), img_PP, kind="cubic")
    newimg_QQ = my_interp_func(xrange(new_shape[0]), xrange(new_shape[1]))
    return newimg_QQ

data_subpath = 'data/fashion'

target_categories = [(0, 'top'), (2, 'pullover'), (6, 'shirt')]

if __name__ == '__main__':
    
    n_keep_per_category_by_split = dict(train=1000, valid=500, test=500)
    P = 20

    for split_name, mnist_name, offset in [
            ('train', 'train', 0),
            ('valid', 'train', 3000),
            ('test', 't10k', 0)]:

        X_train, y_train = mnist_reader.load_mnist(
            os.path.join(FASHION_MNIST_REPO_DIR, data_subpath), kind=mnist_name)

        ## Convert to float between -1.0 and 1.0
        X_train = 2.0 * np.asarray(X_train / 255.0, dtype=np.float32) - 1.0

        keep_x_list = list()
        keep_label_names = list()
        keep_row_ids = list()
        keep_file_names = list()

        for (label_id, label_name) in target_categories:
            print("Constructing split %s for category %s" % (split_name, label_name))
            keep_mask = y_train == label_id
            keep_rows = np.flatnonzero(y_train == label_id)
            cur_x_tr = X_train[keep_mask]
            cur_y_tr = y_train[keep_mask]

            new_x_tr_keep = np.zeros((cur_x_tr.shape[0], P*P))
            kk = 0
            new_x_tr_reject = np.zeros((cur_x_tr.shape[0], P*P))
            rr = 0

            for ii in range(offset, cur_x_tr.shape[0]):
                cur_img_MM = cur_x_tr[ii].reshape((28,28))
                middle_brightness = np.percentile(cur_img_MM[:, 12:16], 60)
                edge_brightness = np.percentile(cur_img_MM[:, :2], 60)

                new_img_PP = interpolate_2d_img(cur_img_MM, (P, P))

                if middle_brightness - edge_brightness > 0.75:
                    new_x_tr_keep[kk] = new_img_PP.reshape((P*P,))
                    keep_label_names.append(label_name)
                    keep_row_ids.append(keep_rows[ii])
                    keep_file_names.append(mnist_name)
                    kk += 1
                else:
                    new_x_tr_reject[rr] = new_img_PP.reshape((P*P,))
                    rr += 1

                if kk >= n_keep_per_category_by_split[split_name]:
                    break
            new_x_tr_keep = new_x_tr_keep[:kk]
            keep_x_list.append(new_x_tr_keep)

            #show_images(new_x_tr_keep, label_name)
            #plt.suptitle("KEEP")
            #show_images(new_x_tr_reject, label_name)
            #plt.suptitle("REJECT")
            #plt.show(block=True)
            #plt.close()

        x_ND = np.vstack(keep_x_list)
        print("    x_ND.shape: %s" % str(x_ND.shape))
        print("    using rows %s ... %s" % (
            ' '.join(['%d' % a for a in keep_row_ids[:5]]),
            ' '.join(['%d' % a for a in keep_row_ids[-5:]]),
            ))
        x_df = pd.DataFrame(x_ND, columns=['pixel%03d' % a for a in range(P*P)])
        x_df.to_csv('tops-%dx%dflattened_x_%s.csv' % (P,P, split_name), index=False, float_format='%.3g')

        y_df = pd.DataFrame()
        y_df['category_name'] = keep_label_names
        y_df['original_file_name'] = keep_file_names
        y_df['original_file_row_id'] = keep_row_ids
        y_df.to_csv('tops-%dx%dflattened_y_%s.csv' % (P,P, split_name), index=False)

        assert x_df.shape[0] == y_df.shape[0]
    #x_te_df = pd.DataFrame(X_test, columns=['pixel%03d' % a for a in range(784)])

    #x_tr_df.to_csv('../x_train.csv', index=False, float_format='%.3g')
    #x_te_df.to_csv('../x_test.csv', index=False, float_format='%.3g')
    
