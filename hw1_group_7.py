import numpy as np
import matplotlib.pyplot as plt
import time
import pdb as bkp


start_time = time.time()  # keep starting time to measure total running time.

# Load hourly arrival rates from the csv file
lamb_v = np.zeros(168)
lamb_m = np.zeros(168)

t = 0
with open('HW1_data.csv', encoding='utf-8-sig') as f:
    for line in f:
        ez = line.strip().split(",")
        lamb_v[t], lamb_m[t] = float(ez[0]), float(ez[1])
        t += 1

# Other Parameters
mu_vv, mu_mm, mu_vm, mu_mv = 14, 13, 9, 10  # The working rates paramteres

L = np.zeros(168)  # Here we will store the expected number of calls in the queue at each hour of the week

# Our code starts here
S = {}
n = 0
queue_tel_dict = np.zeros(168)
for x in range(11):
    for y in range(11):
        for k in [0,"v","m"]:
            for l in [0,"v","m"]:
                S[(x,y,k,l)] = n
                n += 1
x=y=l=k=0
T = 168
P = np.eye(n)
R = np.zeros((n,n))
d1 = 100
d = 1/d1

for i in range (T*d1):
    t = int(i*d)
    if i*d == t: # Checking if we passed a whole hour
        if (t>0 and lamb_m[t]!=lamb_m[t-1] and lamb_v[t]!=lamb_v[t-1]) or t==0:
            # Checking if something changed
            for (x,y,k,l) in S:
                if x==y==0: # If the queues are empty
                    if k==l==0:
                        R[S[(x,y,k,l)],S[(x,y,"v",l)]] = lamb_v[t]
                        R[S[(x,y,k,l)],S[(x,y,k,"m")]] = lamb_m[t]
                    elif k == "v" and l == 0:
                        R[S[(x,y,k,l)],S[(x,y,0,l)]] = mu_vv
                        R[S[(x,y,k,l)],S[(x,y,k,"v")]] = lamb_v[t]
                        R[S[(x,y,k,l)],S[(x,y,k,"m")]] = lamb_m[t]
                    elif k == 0 and l == "m":
                        R[S[(x,y,k,l)],S[(x,y,k,0)]] = mu_mm
                        R[S[(x,y,k,l)],S[(x,y,"v",l)]] = lamb_v[t]
                        R[S[(x,y,k,l)],S[(x,y,"m",l)]] = lamb_m[t]
                    elif k == "m" and l == 0:
                        R[S[(x,y,k,l)],S[(x,y,0,l)]] = mu_vm
                        R[S[(x,y,k,l)],S[(x,y,k,"v")]] = lamb_v[t]
                        R[S[(x,y,k,l)],S[(x,y,k,"m")]] = lamb_m[t]
                    elif k == 0 and l == "v":
                        R[S[(x,y,k,l)],S[(x,y,k,0)]] = mu_mv
                        R[S[(x,y,k,l)],S[(x,y,"v",l)]] = lamb_v[t]
                        R[S[(x,y,k,l)],S[(x,y,"m",l)]] = lamb_m[t]
                    elif k == "v" and l == "m":
                        R[S[(x,y,k,l)],S[(x,y,k,0)]] = mu_mm
                        R[S[(x,y,k,l)],S[(x,y,0,l)]] = mu_vv
                        R[S[(x,y,k,l)],S[(1,y,k,l)]] = lamb_v[t]
                        R[S[(x,y,k,l)],S[(x,1,k,l)]] = lamb_m[t]
                    elif k == "m" and l == "v":
                        R[S[(x,y,k,l)],S[(x,y,k,0)]] = mu_mv
                        R[S[(x,y,k,l)],S[(x,y,0,l)]] = mu_vm
                        R[S[(x,y,k,l)],S[(1,y,k,l)]] = lamb_v[t]
                        R[S[(x,y,k,l)],S[(x,1,k,l)]] = lamb_m[t]
                    elif k == "m" and l == "m":
                        R[S[(x,y,k,l)],S[(x,y,k,0)]] = mu_mm
                        R[S[(x,y,k,l)],S[(x,y,0,l)]] = mu_vm
                        R[S[(x,y,k,l)],S[(1,y,k,l)]] = lamb_v[t]
                        R[S[(x,y,k,l)],S[(x,1,k,l)]] = lamb_m[t]
                    elif k == "v" and l == "v":
                        R[S[(x,y,k,l)],S[(x,y,k,0)]] = mu_mv
                        R[S[(x,y,k,l)],S[(x,y,0,l)]] = mu_vv
                        R[S[(x,y,k,l)],S[(1,y,k,l)]] = lamb_v[t]
                        R[S[(x,y,k,l)],S[(x,1,k,l)]] = lamb_m[t]
                
                elif x > 0 and y == 0: # If the voice inquries queue is not empty
                    if x<10:
                        R[S[(x,y,k,l)],S[(x+1,y,k,l)]] = lamb_v[t]
                    R[S[(x,y,k,l)],S[(x,1,k,l)]] = lamb_m[t]
                    if k == "v" and l == "m":
                        R[S[(x,y,k,l)],S[(x-1,y,k,l)]] = mu_vv
                        R[S[(x,y,k,l)],S[(x-1,y,k,"v")]] = mu_mm
                    elif k == "m" and l == "v":
                        R[S[(x,y,k,l)],S[(x-1,y,k,l)]] = mu_mv
                        R[S[(x,y,k,l)],S[(x-1,y,"v",l)]] = mu_vm
                    elif k == "m" and l == "m":
                        R[S[(x,y,k,l)],S[(x-1,y,k,l)]] = mu_mm
                        R[S[(x,y,k,l)],S[(x-1,y,"v",l)]] = mu_vm
                    elif k == "v" and l == "v":
                        R[S[(x,y,k,l)],S[(x-1,y,k,l)]] = mu_vv + mu_mv
                
                elif x == 0 and y > 0: # If the mail inquries queue is not empty
                    if y<10:
                        R[S[(x,y,k,l)],S[(x,y+1,k,l)]] = lamb_m[t]
                    R[S[(x,y,k,l)],S[(1,y,k,l)]] = lamb_v[t]
                    if k == "v" and l == "m":
                        R[S[(x,y,k,l)],S[(x,y-1,"m",l)]] = mu_vv
                        R[S[(x,y,k,l)],S[(x,y-1,k,l)]] = mu_mm
                    elif k == "m" and l == "v":
                        R[S[(x,y,k,l)],S[(x,y-1,k,"m")]] = mu_mv
                        R[S[(x,y,k,l)],S[(x,y-1,k,l)]] = mu_vm
                    elif k == "m" and l == "m":
                        R[S[(x,y,k,l)],S[(x,y-1,k,l)]] = mu_mm + mu_vm
                    elif k == "v" and l == "v":
                        R[S[(x,y,k,l)],S[(x,y-1,"m",l)]] = mu_vv
                        R[S[(x,y,k,l)],S[(x,y-1,k,"m")]] = mu_mv
                        
                else: # If both queues are not empty
                    if x<10:
                        R[S[(x,y,k,l)],S[(x+1,y,k,l)]] = lamb_v[t]
                    if y<10:
                        R[S[(x,y,k,l)],S[(x,y+1,k,l)]] = lamb_m[t]
                    if k == "v" and l == "m":
                        R[S[(x,y,k,l)],S[(x,y-1,k,l)]] = mu_mm
                        R[S[(x,y,k,l)],S[(x-1,y,k,l)]] = mu_vv
                    elif k == "m" and l == "v":
                        R[S[(x,y,k,l)],S[(x,y-1,k,"m")]] = mu_mv
                        R[S[(x,y,k,l)],S[(x-1,y,"v",l)]] = mu_vm
                    elif k == "m" and l == "v":
                        R[S[(x,y,k,l)],S[(x,y-1,k,"m")]] = mu_mv
                        R[S[(x,y,k,l)],S[(x-1,y,"v",l)]] = mu_vm
                    elif k == "m" and l == "m":
                        R[S[(x,y,k,l)],S[(x,y-1,k,l)]] = mu_mm
                        R[S[(x,y,k,l)],S[(x-1,y,"v",l)]] = mu_vm
                    elif k == "v" and l == "v":
                        R[S[(x,y,k,l)],S[(x,y-1,k,"m")]] = mu_mv
                        R[S[(x,y,k,l)],S[(x-1,y,k,l)]] = mu_vv
    np.fill_diagonal(R, 0)
    np.fill_diagonal(R, -np.sum(R,axis=1))
    PP= np.linalg.matrix_power(np.eye(n)+R*(0.5*d/1024), 1024)
P = np.dot(P,PP)
print("P:")
print(P)
print("R:")
print(R)
# Our code ends here

print(f"Total running time {(time.time() - start_time):.2f} seconds")
print(f"Expected number of calls per week handled by the mail agent: {0}")  # replace 0 with your calculation

plt.style.use("ggplot")
plt.figure(figsize=(10, 6))
plt.plot(range(0, 168), L)
plt.ylabel('Waiting Calls')
plt.xlabel('Hours of the week')
plt.show()