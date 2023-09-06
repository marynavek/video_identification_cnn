import numpy as np

from prnu_extract import extract_single
def bubble_sort(array):

    n = len(array)

    for i in range(n):  
        # Create a flag that will allow the function to
        # terminate early if there's nothing left to sort
        already_sorted = True

        # Start looking at each item of the list one by one,
        # comparing it with its adjacent value. With each
        # iteration, the portion of the array that you look at
        # shrinks because the remaining items have already been
        # sorted.
        # noise_patch = extract_single(img)
        # patch_std = np.std(noise_patch)  
        for j in range(n - i - 1):
            noise_patch_j = extract_single(array[j])
            patch_std_j = np.std(noise_patch_j)
            noise_patch_next = extract_single(array[j+1])
            patch_std_next = np.std(noise_patch_next)
            # if array[j] > array[j + 1]:
            if patch_std_j > patch_std_next:
                array[j], array[j+1] = array[j + 1], array[j]
                # If the item you're looking at is greater than its
                # adjacent value, then swap them
                # array[j], array[j + 1] = array[j + 1], array[j]

                # Since you had to swap two elements,
                # set the `already_sorted` flag to `False` so the
                # algorithm doesn't finish prematurely
                already_sorted = False

        # If there were no swaps during the last iteration,
        # the array is already sorted, and you can terminate
        if already_sorted:
            break
    return array