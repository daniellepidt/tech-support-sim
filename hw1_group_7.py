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
Total_Time_mail_handle_calls = 0

# Our code starts here
S = {} # All the possible situations
max_l = 10 # Maximum queue length 
n = 0
queue_tel_dict = {} # for measure 1 - situations which the voice inquiries queue is not empty
mail_handle_calls = [] # for measure 2 - situations which the mail server handles the call
old_lamb_v = lamb_v[0] +1  # just a dummy value to force calculation at the first itreations
old_lamb_m = lamb_m[0] +1  # just a dummy value to force calculation at the first itreations

# Defining only possible situations in S translate matrix & The relevant situations to the measures
 
# Both queues are empty
for k in [0,"v","m"]:
    for l in [0,"v","m"]: 
        S[(0,0,k,l)] = n
        if l == "v": # for measure 2
            mail_handle_calls.append(n) 
        n += 1
         
# Only one queue is empty
for k in ["v","m"]:
    for l in ["v","m"]: 
        for i in range(1, max_l+1):
            S[(i,0,k,l)] = n
            queue_tel_dict[n] = i # for measure 1
            if l == "v": # for measure 2
                mail_handle_calls.append(n) 
            n += 1
            S[(0,i,k,l)] = n
            if l == "v": # for measure 2
                mail_handle_calls.append(n) 
            n += 1

# Both queues are full
for k in ["v","m"]:
    for l in ["v","m"]: 
        for i in range(1, max_l+1):
            for j in range(1, max_l+1):
                S[(i,j,k,l)] = n
                queue_tel_dict[n] = i # for measure 1
                if l == "v": # for measure 2
                    mail_handle_calls.append(n) 
                n += 1

# Defining the parameters for the simulation
x=y=l=k=0 # We want to use it later
T = 168 # The maximum time of the simulation
P = np.eye(n) 
R = np.zeros((n,n))
d1 = 10 # The amount of time interval
d = 1/d1

# The simulation
for i in range (T*d1):
    t = int(i*d)
    if (old_lamb_v !=  lamb_v[t]) or (old_lamb_m !=  lamb_m[t]):  # Checking if something changed
        for (x,y,k,l) in S:
            if x==y==0: # If the queues are empty, checking the situations R can have
                if k==l==0:
                    R[S[(x,y,k,l)],S[(x,y,"v",l)]] = lamb_v[t]
                    R[S[(x,y,k,l)],S[(x,y,k,"m")]] = lamb_m[t]
                elif k == "v" and l == 0:
                    if R[S[(x,y,k,l)],S[(x,y,0,l)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y,0,l)]] = mu_vv
                    R[S[(x,y,k,l)],S[(x,y,k,"v")]] = lamb_v[t]
                    R[S[(x,y,k,l)],S[(x,y,k,"m")]] = lamb_m[t]
                elif k == 0 and l == "m":
                    if R[S[(x,y,k,l)],S[(x,y,k,0)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y,k,0)]] = mu_mm
                    R[S[(x,y,k,l)],S[(x,y,"v",l)]] = lamb_v[t]
                    R[S[(x,y,k,l)],S[(x,y,"m",l)]] = lamb_m[t]
                elif k == "m" and l == 0:
                    if R[S[(x,y,k,l)],S[(x,y,0,l)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y,0,l)]] = mu_vm
                    R[S[(x,y,k,l)],S[(x,y,k,"v")]] = lamb_v[t]
                    R[S[(x,y,k,l)],S[(x,y,k,"m")]] = lamb_m[t]
                elif k == 0 and l == "v":
                    if R[S[(x,y,k,l)],S[(x,y,k,0)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y,k,0)]] = mu_mv
                    R[S[(x,y,k,l)],S[(x,y,"v",l)]] = lamb_v[t]
                    R[S[(x,y,k,l)],S[(x,y,"m",l)]] = lamb_m[t]
                elif k == "v" and l == "m":
                    if R[S[(x,y,k,l)],S[(x,y,k,0)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y,k,0)]] = mu_mm
                        R[S[(x,y,k,l)],S[(x,y,0,l)]] = mu_vv
                    R[S[(x,y,k,l)],S[(1,y,k,l)]] = lamb_v[t]
                    R[S[(x,y,k,l)],S[(x,1,k,l)]] = lamb_m[t]
                elif k == "m" and l == "v":
                    if R[S[(x,y,k,l)],S[(x,y,k,0)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y,k,0)]] = mu_mv
                        R[S[(x,y,k,l)],S[(x,y,0,l)]] = mu_vm
                    R[S[(x,y,k,l)],S[(1,y,k,l)]] = lamb_v[t]
                    R[S[(x,y,k,l)],S[(x,1,k,l)]] = lamb_m[t]
                elif k == "m" and l == "m":
                    if R[S[(x,y,k,l)],S[(x,y,k,0)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y,k,0)]] = mu_mm
                        R[S[(x,y,k,l)],S[(x,y,0,l)]] = mu_vm
                    R[S[(x,y,k,l)],S[(1,y,k,l)]] = lamb_v[t]
                    R[S[(x,y,k,l)],S[(x,1,k,l)]] = lamb_m[t]
                elif k == "v" and l == "v":
                    if R[S[(x,y,k,l)],S[(x,y,k,0)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y,k,0)]] = mu_mv 
                        R[S[(x,y,k,l)],S[(x,y,0,l)]] = mu_vv
                    R[S[(x,y,k,l)],S[(1,y,k,l)]] = lamb_v[t]
                    R[S[(x,y,k,l)],S[(x,1,k,l)]] = lamb_m[t]
            
            elif x > 0 and y == 0: # If the voice inquiries queue is not empty
                if x<max_l:
                    R[S[(x,y,k,l)],S[(x+1,y,k,l)]] = lamb_v[t]
                R[S[(x,y,k,l)],S[(x,1,k,l)]] = lamb_m[t]
                if k == "v" and l == "m":
                    if R[S[(x,y,k,l)],S[(x-1,y,k,l)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x-1,y,k,l)]] = mu_vv
                        R[S[(x,y,k,l)],S[(x-1,y,k,"v")]] = mu_mm
                elif k == "m" and l == "v":
                    if R[S[(x,y,k,l)],S[(x-1,y,k,l)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x-1,y,k,l)]] = mu_mv
                        R[S[(x,y,k,l)],S[(x-1,y,"v",l)]] = mu_vm
                elif k == "m" and l == "m":
                    if R[S[(x,y,k,l)],S[(x-1,y,k,"v")]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x-1,y,k,"v")]] = mu_mm
                        R[S[(x,y,k,l)],S[(x-1,y,"v",l)]] = mu_vm
                elif k == "v" and l == "v":
                    if R[S[(x,y,k,l)],S[(x-1,y,k,l)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x-1,y,k,l)]] = mu_vv + mu_mv
            
            elif x == 0 and y > 0: # If the mail inquiries queue is not empty
                if y<max_l:
                    R[S[(x,y,k,l)],S[(x,y+1,k,l)]] = lamb_m[t]
                R[S[(x,y,k,l)],S[(1,y,k,l)]] = lamb_v[t]
                if k == "v" and l == "m":
                    if R[S[(x,y,k,l)],S[(x,y-1,"m",l)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y-1,"m",l)]] = mu_vv
                        R[S[(x,y,k,l)],S[(x,y-1,k,l)]] = mu_mm
                elif k == "m" and l == "v":
                    if R[S[(x,y,k,l)],S[(x,y-1,"m",l)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y-1,k,"m")]] = mu_mv
                        R[S[(x,y,k,l)],S[(x,y-1,k,l)]] = mu_vm
                elif k == "m" and l == "m":
                    if R[S[(x,y,k,l)],S[(x,y-1,k,l)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y-1,k,l)]] = mu_mm + mu_vm
                elif k == "v" and l == "v":
                    if R[S[(x,y,k,l)],S[(x,y-1,"m",l)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y-1,"m",l)]] = mu_vv
                        R[S[(x,y,k,l)],S[(x,y-1,k,"m")]] = mu_mv
                    
            else: # If both queues are not empty
                if x<max_l:
                    R[S[(x,y,k,l)],S[(x+1,y,k,l)]] = lamb_v[t]
                if y<max_l:
                    R[S[(x,y,k,l)],S[(x,y+1,k,l)]] = lamb_m[t]
                if k == "v" and l == "m":
                    if R[S[(x,y,k,l)],S[(x,y-1,k,l)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y-1,k,l)]] = mu_mm
                        R[S[(x,y,k,l)],S[(x-1,y,k,l)]] = mu_vv
                elif k == "m" and l == "v":
                    if R[S[(x,y,k,l)],S[(x,y-1,k,"m")]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y-1,k,"m")]] = mu_mv
                        R[S[(x,y,k,l)],S[(x-1,y,"v",l)]] = mu_vm
                elif k == "m" and l == "m":
                    if R[S[(x,y,k,l)],S[(x,y-1,k,l)]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y-1,k,l)]] = mu_mm
                        R[S[(x,y,k,l)],S[(x-1,y,"v",l)]] = mu_vm
                elif k == "v" and l == "v":
                    if R[S[(x,y,k,l)],S[(x,y-1,k,"m")]] == 0: # Updating mu happens only once, because it is time independent
                        R[S[(x,y,k,l)],S[(x,y-1,k,"m")]] = mu_mv
                        R[S[(x,y,k,l)],S[(x-1,y,k,l)]] = mu_vv
        
        np.fill_diagonal(R, 0)
        np.fill_diagonal(R, -np.sum(R,axis=1))
        PP= np.linalg.matrix_power(np.eye(n)+R*(0.5*d/1024), 1024)
    
    P = np.dot(P,PP)
    
    # Updating measure 2 
    for s,l in queue_tel_dict.items():
        L[t] += P[0,s] * l * d
    # Updating measure 3
    for s in mail_handle_calls:
        Total_Time_mail_handle_calls += P[0,s] * 1 * d
    
    P = np.dot(P,PP)
    old_lamb_v = lamb_v[t]  # just a dummy value to force calculation at the first itreations
    old_lamb_m = lamb_m[t]  # just a dummy value to force calculation at the first itreations

# Our code ends here

print(f"Total running time {(time.time() - start_time):.2f} seconds")
print(f"Expected number of calls per week handled by the mail agent: {Total_Time_mail_handle_calls*mu_mv}")  # replace 0 with your calculation

plt.style.use("ggplot")
plt.figure(figsize=(10, 6))
plt.plot(range(0, 168), L)
plt.ylabel('Waiting Calls')
plt.xlabel('Hours of the week')
plt.show()