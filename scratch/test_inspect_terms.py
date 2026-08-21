import sys, os
sys.path.insert(0, os.path.abspath('.'))
import pulp
import numpy as np
from algorithms.worker_manager import generate_worker_profiles

nw = 24
nd = 7
rd = 4
re = 4
rn = 4
weights = {'circadian': 50, 'night_imb': 25, 'workload_imb': 20, 'exp_mix': 60, 'pref_off': 40, 'posta': 15}
workers = generate_worker_profiles(nw, nd, randomize=False)

w_night = weights.get('night_imb', 25)
w_work = weights.get('workload_imb', 20)
w_exp = weights.get('exp_mix', 60)
w_pref = weights.get('pref_off', 40)
w_posta = weights.get('posta', 15)
w_circ = weights.get('circadian', 50)

model = pulp.LpProblem("LP_Relaxation_Bound", pulp.LpMinimize)
shifts = [0, 1, 2, 3]
shift_reqs = {1: rd, 2: re, 3: rn}
postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
ustas_wids = [w['id'] for w in workers if w['is_usta']]

x = pulp.LpVariable.dicts("x", ((i, t, k) for i in range(nw) for t in range(nd) for k in shifts), cat=pulp.LpContinuous, lowBound=0, upBound=1)
sirk = pulp.LpVariable.dicts("sirkadiyen", ((i, t) for i in range(nw) for t in range(nd - 1)), cat=pulp.LpContinuous, lowBound=0)
pref_viol = pulp.LpVariable.dicts("pref_viol", (i for i in range(nw)), cat=pulp.LpContinuous, lowBound=0)
y_posta = pulp.LpVariable.dicts("y_posta", ((p, t, k) for p in postas for t in range(nd) for k in [1, 2, 3]), lowBound=0, cat=pulp.LpContinuous)
z_posta = pulp.LpVariable.dicts("z_posta", ((p, t, k) for p in postas for t in range(nd) for k in [1, 2, 3]), cat=pulp.LpContinuous, lowBound=0, upBound=1)
dev_posta_k = pulp.LpVariable.dicts("dev_posta_k", ((p, t, k) for p in postas for t in range(nd) for k in [1, 2, 3]), lowBound=0, cat=pulp.LpContinuous)
posta_dev = pulp.LpVariable.dicts("posta_dev", ((p, t) for p in postas for t in range(nd)), lowBound=0, cat=pulp.LpContinuous)
no_usta = pulp.LpVariable.dicts("no_usta", ((t, k) for t in range(nd) for k in [1, 2, 3]), cat=pulp.LpContinuous, lowBound=0, upBound=1)

d_pos = pulp.LpVariable.dicts("d_pos", (i for i in range(nw)), lowBound=0, cat=pulp.LpContinuous)
d_neg = pulp.LpVariable.dicts("d_neg", (i for i in range(nw)), lowBound=0, cat=pulp.LpContinuous)

d_pos_work = pulp.LpVariable.dicts("d_pos_work", (i for i in range(nw)), lowBound=0, cat=pulp.LpContinuous)
d_neg_work = pulp.LpVariable.dicts("d_neg_work", (i for i in range(nw)), lowBound=0, cat=pulp.LpContinuous)

target_avg_night = (rn * nd) / max(1, nw)
target_avg_work = ((rd + re + rn) * nd) / max(1, nw)

for i in range(nw):
    for t in range(nd):
        model += pulp.lpSum([x[i, t, k] for k in shifts]) == 1
        
for t in range(nd):
    for k in [1, 2, 3]:
        model += pulp.lpSum([x[i, t, k] for i in range(nw)]) == shift_reqs[k]
        
req_certs = ['Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu']
for cert in req_certs:
    cert_wids = [w['id'] for w in workers if cert in w['skills']]
    if len(cert_wids) > 0:
        for t in range(nd):
            for k in [1, 2, 3]:
                model += pulp.lpSum([x[i, t, k] for i in cert_wids]) >= 1
            
for i in range(nw):
    for t in range(nd - 1):
        model += x[i, t, 3] + x[i, t+1, 1] <= 1
        model += sirk[i, t] >= x[i, t, 2] + x[i, t+1, 1] - 1
        
for i in range(nw):
    for t in range(nd - 6):
        model += pulp.lpSum([x[i, tau, 0] for tau in range(t, t + 7)]) >= 1

for i in range(nw):
    p_day_idx = int(workers[i]['pref_off']) - 1
    if 0 <= p_day_idx < nd:
        model += pref_viol[i] >= 1 - x[i, p_day_idx, 0]

for t in range(nd):
    for k in [1, 2, 3]:
        if len(ustas_wids) > 0:
            model += no_usta[t, k] >= 1 - pulp.lpSum([x[i, t, k] for i in ustas_wids])
        else:
            model += no_usta[t, k] == 1

for i in range(nw):
    night_sum_i = pulp.lpSum([x[i, t, 3] for t in range(nd)])
    model += night_sum_i - target_avg_night == d_pos[i] - d_neg[i]

for i in range(nw):
    work_sum_i = pulp.lpSum([x[i, t, k] for t in range(nd) for k in [1, 2, 3]])
    model += work_sum_i - target_avg_work == d_pos_work[i] - d_neg_work[i]

for t in range(nd):
    for p in postas:
        p_wids = [w['id'] for w in workers if w['posta'] == p]
        N_p = len(p_wids)
        model += pulp.lpSum([z_posta[p, t, k] for k in [1, 2, 3]]) <= 1
        for k in [1, 2, 3]:
            model += y_posta[p, t, k] == pulp.lpSum([x[wid, t, k] for wid in p_wids])
            model += dev_posta_k[p, t, k] >= y_posta[p, t, k] - N_p * z_posta[p, t, k]
        model += posta_dev[p, t] == pulp.lpSum([dev_posta_k[p, t, k] for k in [1, 2, 3]])

term_circ = w_circ * pulp.lpSum([sirk[i, t] for i in range(nw) for t in range(nd - 1)])
term_pref = w_pref * pulp.lpSum([pref_viol[i] for i in range(nw)])
term_posta = w_posta * pulp.lpSum([posta_dev[p, t] for p in postas for t in range(nd)])
term_exp = w_exp * pulp.lpSum([no_usta[t, k] for t in range(nd) for k in [1, 2, 3]])
term_night = w_night * pulp.lpSum([d_pos[i] + d_neg[i] for i in range(nw)])
term_work = w_work * pulp.lpSum([d_pos_work[i] + d_neg_work[i] for i in range(nw)])

total_penalty_expr = term_circ + term_pref + term_posta + term_exp + term_night + term_work
model += total_penalty_expr
solver = pulp.PULP_CBC_CMD(msg=False)
model.solve(solver)

print("model status:", model.status)
print("model objective:", pulp.value(model.objective))
print("term_circ:", pulp.value(term_circ))
print("term_pref:", pulp.value(term_pref))
print("term_posta:", pulp.value(term_posta))
print("term_exp:", pulp.value(term_exp))
print("term_night:", pulp.value(term_night))
print("term_work:", pulp.value(term_work))
print("Sum of terms:", pulp.value(term_circ) + pulp.value(term_pref) + pulp.value(term_posta) + pulp.value(term_exp) + pulp.value(term_night) + pulp.value(term_work))
