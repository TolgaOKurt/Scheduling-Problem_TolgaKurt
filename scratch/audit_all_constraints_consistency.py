import os, re, sys
sys.stdout.reconfigure(encoding='utf-8')

def check_files():
    algorithms = [
        'algorithms/penalty_calculator.py',
        'algorithms/ilp_pulp_solver.py',
        'algorithms/csp_backtracking_solver.py',
        'algorithms/greedy_solver.py',
        'algorithms/hill_climbing_solver.py',
        'algorithms/simulated_annealing_solver.py',
        'algorithms/genetic_algorithm_solver.py',
        'algorithms/memetic_algorithm_solver.py',
        'algorithms/tabu_search_solver.py',
        'algorithms/vns_solver.py',
        'algorithms/pso_solver.py',
        'algorithms/aco_solver.py',
        'algorithms/worker_manager.py',
        'algorithms/evolutionary_engine.py',
        'algorithms/solver_contract.py',
        'global_state.py',
        'views/tab3_steel_model.py',
        'views/tab6_ilp_optimization.py',
        'views/tab5_csp_backtracking.py',
        'views/common_components.py'
    ]
    
    print("=" * 80)
    print(" KISIT VE CEZA PUANI TUTARLILIK DENETİMİ")
    print("=" * 80)
    
    for filepath in algorithms:
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        print(f"\n📁 {filepath}:")
        # Check hard constraint references
        hard_refs = re.findall(r'(?:Sert Kısıt|Hard Constraint|hard_violation|audit_all_hard_constraints|check_swap_feasibility)', content, re.IGNORECASE)
        print(f"   - Sert Kısıt Referansları: {len(hard_refs)} adet")
        
        # Check soft weights references
        weights_found = set()
        for w in ['circadian', 'night_imb', 'exp_mix', 'pref_off', 'posta', 'posta_unity']:
            if w in content:
                weights_found.add(w)
        print(f"   - Kullanılan Ceza Ağırlığı Anahtarları: {sorted(list(weights_found))}")

if __name__ == '__main__':
    check_files()
