def calc_sol_delta(arr, split):
    total_delta = 0
    for i in range(len(arr)):
        ar = arr[i]
        sum1 = sum(ar)
        cnt1 = sum([ar[ix] for ix in split])
        total_delta += abs(sum1 - 2*cnt1)
    return total_delta
