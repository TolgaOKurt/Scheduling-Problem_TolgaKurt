import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from algorithms.worker_manager import generate_worker_profiles
from algorithms.cp_sat_solver import solve_cp_sat
from algorithms.ilp_pulp_solver import solve_ilp_pulp
from algorithms.greedy_solver import run_greedy_algorithm

n_workers, n_days, r_day, r_eve, r_night = 28, 7, 9, 7, 5
weights = {'circadian': 50, 'night_imb': 25, 'exp_mix': 30, 'pref_off': 40, 'posta': 15}
workers = generate_worker_profiles(n_workers, n_days, randomize=False)

print("1. Testing Google CP-SAT solver execution...")
res_cpsat = solve_cp_sat(
    n_workers, n_days, r_day, r_eve, r_night, weights,
    time_limit=5.0, num_threads=8, custom_workers=workers
)

# Kontrat Doğrulaması
required_keys = [
    'schedule', 'workers', 'is_feasible', 'hard_violations_count',
    'hard_violation_logs', 'final_score', 'initial_score', 'improvement_rate',
    'exec_time_ms', 'eval_count', 'total_iterations', 'termination_reason',
    'penalties', 'request_details', 'score_history', 'meta'
]
for k in required_keys:
    assert k in res_cpsat, f"Eksik anahtar: {k}"

print(f"✅ Kontrat Doğrulandı! (Tüm {len(required_keys)} alan mevcut)")
print(f"   - Geçerlilik: {res_cpsat['is_feasible']}")
print(f"   - Sert İhlal Sayısı: {res_cpsat['hard_violations_count']}")
print(f"   - Final Skor (Z): {res_cpsat['final_score']}")
print(f"   - Çözüm Süresi: {res_cpsat['exec_time_ms']} ms ({res_cpsat['meta']['wall_time']} sn)")
print(f"   - Çözücü Durumu: {res_cpsat['meta']['solver_status']}")
print(f"   - MIP Gap: %{res_cpsat['meta']['mip_gap']}")
print(f"   - Ceza Dağılımı: {res_cpsat['penalties']}")

# Toplam ceza eşitliği kontrolü
calc_sum = sum(res_cpsat['penalties'].values())
assert calc_sum == res_cpsat['final_score'], f"Ceza toplamı uyuşmuyor: {calc_sum} != {res_cpsat['final_score']}"
print(f"✅ Ceza Puanı Matematiksel Sağlaması Başarılı: {calc_sum} == {res_cpsat['final_score']}")

print("\n2. CP-SAT vs ILP vs Greedy Hızlı Karşılaştırma:")
res_ilp = solve_ilp_pulp(n_workers, n_days, r_day, r_eve, r_night, weights, time_limit=5, custom_workers=workers)
res_greedy = run_greedy_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=workers)

print(f"{'Yöntem':<25} | {'Feasible':<8} | {'İhlal':<5} | {'Skor (Z)':<8} | {'Süre (ms)':<10}")
print("-" * 65)
print(f"{'Greedy (Yapıcı)':<25} | {str(res_greedy['is_feasible']):<8} | {res_greedy['hard_violations_count']:>5} | {res_greedy['final_score']:>8} | {res_greedy['exec_time_ms']:>10}")
print(f"{'ILP / PuLP (CBC)':<25} | {str(res_ilp['is_feasible']):<8} | {res_ilp['hard_violations_count']:>5} | {res_ilp['final_score']:>8} | {res_ilp['exec_time_ms']:>10}")
print(f"{'Google CP-SAT (OR-Tools)':<25} | {str(res_cpsat['is_feasible']):<8} | {res_cpsat['hard_violations_count']:>5} | {res_cpsat['final_score']:>8} | {res_cpsat['exec_time_ms']:>10}")
print("-" * 65)
