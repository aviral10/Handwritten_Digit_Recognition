n = 256
arr = []
for i in range(1, 128+1):
    if n%i == 0:
        arr.append([i, n//i])
print(arr)