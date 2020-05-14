import numpy as np

def im2col(input_data, filter_h, filter_w, stride=2, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 1, 4, 5, 2, 3).reshape(N*C*out_h*out_w, -1)

    return col

def pooling_layer(X, amount):

    col_X = im2col(X, 2, 2)

    '''sp_col = np.split(col_X, 4, 1)

    col1_max = np.max(sp_col[0], axis=1)
    col2_max = np.max(sp_col[1], axis=1)
    col3_max = np.max(sp_col[2], axis=1)
    col4_max = np.max(sp_col[3], axis=1)

    col1_max = col1_max.reshape(amount, 16, 16)
    col2_max = col2_max.reshape(amount, 16, 16)
    col3_max = col3_max.reshape(amount, 16, 16)
    col4_max = col4_max.reshape(amount, 16, 16)

    pooled_X = np.concatenate([np.concatenate([col1_max, col2_max], 2), np.concatenate([col3_max, col4_max], 2)], 1)'''

    pooled_X = np.max(col_X, axis=1)
    pooled_X = pooled_X.reshape(amount, 32, 32)
 
    return pooled_X

def pooling_layer2(X, amount):

    col_X = im2col(X, 2, 2)

    pooled_X = np.max(col_X, axis=1)
    pooled_X = pooled_X.reshape(amount, -1)
 
    return pooled_X
