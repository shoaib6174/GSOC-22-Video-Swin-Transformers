import tensorflow as tf
def get_window_size(x_size, window_size, shift_size=None):
    #print("get window",x_size, window_size, shift_size)

    # print("get_window_size parameters",x_size, window_size, shift_size)

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0



    if shift_size is None:
        return tuple(use_window_size)
    else:
        # print(tuple(use_window_size), tuple(use_shift_size))
        return tuple(use_window_size), tuple(use_shift_size)