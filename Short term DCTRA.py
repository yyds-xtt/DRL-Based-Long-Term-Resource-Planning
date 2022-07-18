import math
import random
def bubbleSort(alist):
    for passnum in range(len(alist)-1, 0, -1): # 5, 4~1
        for i in range(passnum): # 每次只需要前面的比较，最后一位是每一轮最大的
            if alist[i] > alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp
    return alist  # 不要忘记 return，否则前面的排序过程没有输出，就没有意义了

def getReward(req, f_0):
    Vertex_num_mMTC = req[1]
    Vertex_num_uRLLC = req[2]
    uRLLC = []
    mMTC = []
    x = 0
    y = 0
    while x < Vertex_num_uRLLC:
        req_comp = random.randrange(25, 28)
        local_capability = random.randrange(4, 6)
        uRLLC.append([req_comp, local_capability, 5 / 6, 1 / 6])  # 所有的请求信息
        x += 1
    while y < Vertex_num_mMTC:
        req_comp = random.randrange(10, 14)
        local_capability = random.randrange(2, 4)
        mMTC.append([req_comp, local_capability, 1 / 6, 5 / 6])  # 所有的请求信息
        y += 1

    total = []
    for i in uRLLC:
        total.append(i)
    for i in mMTC:
        total.append(i)

    h_i = 1
    N_0 = 0.01
    alpha = 0.1
    Xi = 1
    p_max = 10
    bw = 3

    for i in range(len(total)-1, 0, -1):
        for ii in range(i):
            if total[ii][1]/total[ii][2] > total[ii + 1][1]/total[ii+1][2]:
                temp = total[ii]
                total[ii] = total[ii + 1]
                total[ii + 1] = temp

    # 计算所有的pi # Recursion # iteration
    total_p = []
    total_tran_time = []
    total_remote_power = []
    total_local_time = []
    total_local_energy = []


    for i in total:
        Eta = i[2]*i[0]/(bw*(i[0]/i[1]))
        Gamma = (1-i[2])*i[0]/(bw*Xi*(alpha*i[1]*i[0]))
        Phi = Gamma*(math.log(1 + h_i*p_max/N_0, 2)) - ((h_i/N_0)/(math.log(2, math.e))) * (Eta + Gamma*p_max)/(1 + h_i*p_max/N_0)
        p = 0
        if Phi <= 0:
            p = p_max
        else:
            p_s = 0
            p_t = p_max
            p_l = (p_t + p_s)/2
            check = 0
            while p_t - p_s >= 0.3:
                if Gamma*(math.log(1 + h_i*p_l/N_0, 2)) - ((h_i/N_0)/(math.log(2, math.e))) * (Eta + Gamma*p_l)/(1 + h_i*p_l/N_0) <= 0:
                    p = p_l
                    check = 1
                    break
                else:
                    p_t = p_l
                    p_l = (p_t + p_s) / 2
            if check == 0:
                p = (p_t + p_s)/2
        local_energy = alpha * i[0] * i[1]
        local_time = i[0] / i[1]
        tran_time = i[0] / (bw * (math.log(1+h_i*p/N_0, 2)))
        remote_power = p/Xi * tran_time
        total_tran_time.append(tran_time)
        total_remote_power.append(remote_power)
        total_p.append(p)
        total_local_time.append(local_time)
        total_local_energy.append(local_energy)

    # if num == 1 or num == 2:
        ## Offloading decision ##
    reward = 0
    time = 0
    rejected = []
    i = 0
    while i < len(total):
        if total[i][2] == 5/6:
            time = total[i][0] / f_0 + time
            if time >= total_local_time[i]:
                reward += 0
                rejected.append(i)
                i += 1
            else:
                log = 5/6 * (total_local_time[i] - (total_tran_time[i] + time)) / total_local_time[i] + 1/6 * (total_local_energy[i] - total_remote_power[i]) / total_local_energy[i]
                if log < 0:
                    reward += 0
                    rejected.append(i)
                    i += 1
                else:
                    reward += log
                    i += 1

        else: # i[2] == 1/6
            time = total[i][0] / f_0 + time
            if total_remote_power[i] >= total_local_energy[i]:
                reward += 0
                rejected.append(i)
                i += 1
            else:
                log = 1/6 * (total_local_time[i] - (total_tran_time[i] + time)) / total_local_time[i] + 5/6 * (total_local_energy[i] - total_remote_power[i]) / total_local_energy[i]
                if log < 0:
                    reward += 0
                    rejected.append(i)
                    i += 1
                else:
                    reward += log
                    i += 1
    reward = reward/len(total)
    occupation_time = time

    return reward, occupation_time