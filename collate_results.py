import math
import os
import json
from statistics import mean
import matplotlib.pyplot as plt

params = {'legend.fontsize': 'x-large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

def find_lines_with_string(lines, target_string):
    matching_lines = []
    
    for i, line in enumerate(lines):
        if target_string in line:
            matching_lines.append(lines[i][lines[i].rfind(" ") + 1:])
    return matching_lines

def find_lines_with_string_list(lines, target_string):
    matching_lines = []
    
    for i, line in enumerate(lines):
        if target_string in line:
            matching_lines.append(lines[i][lines[i].rfind("["):])

    return matching_lines

domains = {
    'blocks': 'results/blocks/with-presolve',
    'gripper': 'results/gripper/with-presolve',
    'n-puzzle': 'results/npuzzle/with-presolve',
    'parking': 'results/parking/with-presolve',
}    

naive_domains = {
    'blocks': 'results/blocks/naive_test',
    'gripper': 'results/gripper/naive_test',
    'n-puzzle': 'results/npuzzle/naive_test',
    'parking': 'results/parking/naive_test',
}  

# Log of the runtimes of each problem
runtime = {}
runtime_ordering = {}
# Do size as the number of inputs + inner modules + 1 (output)
size = {}
# Size of input
size_input = {}
# Size of explanation
size_exp = {}
size_exp_ordering = {}
# max and min of the gurobi runtimes for each step
gurobi_runtimes = {}
# Number of calls/plan length
calls_over_length = {}

main_length = {}
naive_length = {}

main_length_runtimes = {}
naive_length_runtimes = {}
cumulative = {
    "Naive": [],
    "Alg1": []
}
naive_runtimes = []
main_runtimes = []

problems_solved = []

unsolved = {}

for domain in domains:
    for info in [runtime, runtime_ordering, size, size_input, size_exp, size_exp_ordering, gurobi_runtimes, calls_over_length]:
        info[domain] = []   

# Extract info
for domain, path in naive_domains.items():
    # print("Unsolved:", domain, unsolved[domain])
    sorted_files = sorted([f for f in os.listdir(path) if os.path.isfile(path + "/" + f)])
    for log_file in sorted_files:
        with open(path + "/" + log_file, "r") as f:           
            lines = f.read().split("\n")
            
            goal_reached = int(find_lines_with_string(lines, "Goal reached/Finished in time")[0])
            non_trivial_traj = int(find_lines_with_string(lines, "Plan length:")[0])
            if goal_reached and non_trivial_traj >= 1:
                problems_solved.append(lines[0][lines[0].rfind("/"):])
                
                time_secs = float(find_lines_with_string(lines, ":Runtime: ")[0])
                naive_runtimes.append(time_secs)
                            
                if non_trivial_traj in naive_length:
                    naive_length[non_trivial_traj] += 1
                    naive_length_runtimes[non_trivial_traj].append(float(find_lines_with_string(lines, ":Runtime: ")[0]))
                else:
                    naive_length[non_trivial_traj] = 1
                    naive_length_runtimes[non_trivial_traj] = [float(find_lines_with_string(lines, ":Runtime: ")[0])]
                    
cumulative_coverage = 0
cumulative_runtime = 0
cumulative["Naive"]=[(0,0)]
naive_runtimes.sort()
for run in naive_runtimes:
    cumulative_coverage += 1
    cumulative_runtime +=run
    cumulative["Naive"].append((cumulative_runtime, cumulative_coverage))
naive_max_runtime = cumulative_runtime
naive_max_coverage = cumulative_coverage

for domain, path in domains.items():
    unsolved[domain] = 0
    sorted_files = sorted([f for f in os.listdir(path) if os.path.isfile(path + "/" + f)])
    for log_file in sorted_files:
        with open(path + "/" + log_file, "r") as f:           
            lines = f.read().split("\n")
            
            goal_reached = int(find_lines_with_string(lines, "Goal reached/Finished in time")[0])
            non_trivial_traj = int(find_lines_with_string(lines, "Plan length:")[0])
            if goal_reached and non_trivial_traj >= 1:
                time_secs = float(find_lines_with_string(lines, ":Runtime: ")[1])
                runtime[domain].append(time_secs)
                main_runtimes.append(time_secs)
                runtime_ordering[domain].append([float(find_lines_with_string(lines, ":Runtime: ")[i]) for i in range(2)])
                
                prob_input = int(find_lines_with_string(lines, "Size of input")[1])
                size_input[domain].append(prob_input)
                
                prob_exp = int(find_lines_with_string(lines, "Size of explanation:")[1])
                size_exp[domain].append(prob_exp / prob_input)
                size_exp_ordering[domain].append([int(find_lines_with_string(lines, "Size of explanation:")[i]) / prob_input for i in range(2)])
                
                size[domain].append(int(find_lines_with_string(lines, "Number of inner modules ASNets")[1]))
                
                gurobi_times = json.loads(find_lines_with_string_list(lines, "Runtime for each network call:")[1])
                gurobi_runtimes[domain].append((min(gurobi_times), max(gurobi_times), mean(gurobi_times)))
                
                if int(find_lines_with_string(lines, "Plan length:")[1]) > 1:
                    call_over_length = (int(find_lines_with_string(lines, "Number of calls:")[1]) / prob_input) / \
                        int(find_lines_with_string(lines, "Plan length:")[1])
                    calls_over_length[domain].append(call_over_length)
                
                if int(find_lines_with_string(lines, "Plan length:")[1]) in main_length:
                    main_length[int(find_lines_with_string(lines, "Plan length:")[1])] += 1
                    if lines[0][lines[0].rfind("/"):] in problems_solved:
                        main_length_runtimes[int(find_lines_with_string(lines, "Plan length:")[0])].append(float(find_lines_with_string(lines, ":Runtime: ")[1]))
                else:
                    main_length[int(find_lines_with_string(lines, "Plan length:")[1])] = 1
                    if lines[0][lines[0].rfind("/"):] in problems_solved:
                        main_length_runtimes[int(find_lines_with_string(lines, "Plan length:")[0])] = [float(find_lines_with_string(lines, ":Runtime: ")[1])]
            elif not goal_reached:
                unsolved[domain] += 1
                
cumulative_coverage = 0
cumulative_runtime = 0
cumulative["Alg1"] = [(0,0)]
main_runtimes.sort()                   
for run in main_runtimes:
    cumulative_coverage += 1
    cumulative_runtime += run
    cumulative["Alg1"].append((cumulative_runtime, cumulative_coverage))
main_max_runtime = cumulative_runtime
main_max_coverage = cumulative_coverage

max_runtime = max(main_max_runtime,naive_max_runtime)*1.5
cumulative["Naive"].append((max_runtime,naive_max_coverage))
cumulative["Alg1"].append((max_runtime,main_max_coverage))

    
# Cumulative graph
plt.figure()

for alg, data in cumulative.items():
    x, y = zip(*data)
    plt.plot(x, y, label=alg)
    
plt.xscale('log')
plt.xlabel('Cumulative Run Time (seconds)')
plt.ylabel('Cumulative Coverage')
plt.legend()
    
plt.show()
                    
# Box for number of problems solved by traj length
plt.figure()
plt.bar(main_length.keys(), main_length.values(), width=0.35, label="Algorithm 1")
plt.bar([k+0.35 for k in naive_length.keys()], naive_length.values(), width=0.35, label='Naive Algorithm')
plt.legend()
plt.xlabel("Length of the trajectory")
plt.ylabel("Number of problems solved")
plt.savefig("graphs/alg_1_vs_naive.pdf", format='pdf')

means_main = {}
for k,v in main_length_runtimes.items():
    if k in naive_length_runtimes:
        means_main[k] = mean(v)
means_naive = {}
for k,v in naive_length_runtimes.items():
    means_naive[k] = mean(v)
plt.figure()
plt.bar(means_main.keys(), means_main.values(), width=0.35, label="Algorithm 1")
plt.bar([k+0.35 for k in means_naive.keys()], means_naive.values(), width=0.35, label='Naive Algorithm')
plt.legend()
plt.xlabel("Length of the trajectory")
plt.ylabel("Average runtime (seconds)")
plt.yscale("log")
plt.savefig("graphs/alg_1_vs_naive_runtime.pdf", format='pdf')

# Plot for size vs log runtime    
plt.figure()
markers = ["X" , "," , "o" , "v" , "^" , "<", ">"]
m = 0 
for domain in domains.keys():
    plt.scatter(size[domain], runtime[domain], marker=markers[m], label=domain)
    m += 1
    
plt.legend()
#plt.xlabel("Number of modules in the neural network")
plt.xlabel("Nb modules in neural network")
#plt.ylabel("Runtime (seconds)")
plt.ylabel("Runtime (sec)")
plt.yscale('log')
plt.savefig("graphs/runtime_over_size.pdf", format='pdf')

# Compare runtime for both orderings
plt.figure()
fig, ax = plt.subplots()
ax.plot([0,1],[0,1], color='k', transform=ax.transAxes, zorder=0)
markers = ["X" , "," , "o" , "v" , "^" , "<", ">"]
m = 0
for domain in domains.keys():
    least_to_most = []
    most_to_least = []
    for problem in runtime_ordering[domain]:
        least_to_most.append(problem[0])
        most_to_least.append(problem[1])
    plt.scatter(least_to_most, most_to_least, marker=markers[m], label=domain)
    m += 1
    
plt.legend()
#plt.xlabel("Runtime (seconds) [Least -> Most sensitive]")
plt.xlabel("Runtime [Least -> Most sensitive]")
#plt.ylabel("Runtime (seconds) [Most -> Least sensitive]")
plt.ylabel("Runtime [Most -> Least sensitive]")
plt.xscale("log")
plt.yscale("log")
plt.savefig("graphs/ordering_runtime.pdf", format='pdf')

# Compare explanation size for both orderings
plt.figure()
fig, ax = plt.subplots()
ax.plot([0,1],[0,1], color='k', transform=ax.transAxes, zorder=0)
markers = ["X" , "," , "o" , "v" , "^" , "<", ">"]
m = 0
for domain in domains.keys():
    least_to_most = []
    most_to_least = []
    for problem in size_exp_ordering[domain]:
        least_to_most.append(problem[0])
        most_to_least.append(problem[1])
    plt.scatter(least_to_most, most_to_least, marker=markers[m], label=domain)
    m += 1
    
plt.legend()
#plt.xlabel("Percentage of input [Least -> Most sensitive]")
plt.xlabel("Expl. size [Least -> Most sensitive]")
#plt.ylabel("Percentage of input [Most -> Least sensitive]")
plt.ylabel("Expl. size [Most -> Least sensitive]")
plt.savefig("graphs/ordering_exp_size.pdf", format='pdf')

# Plot size explanation vs size of input
plt.figure()
markers = ["X" , "," , "o" , "v" , "^" , "<", ">"]
m = 0 
for domain in domains.keys():
    plt.scatter(size_input[domain], size_exp[domain], marker=markers[m], label=domain)
    m += 1
    
plt.legend()
#plt.xlabel("Number of propositions in the input")
plt.xlabel("Nb propositions in input")
#plt.ylabel("Percentage of the input in the explanation")
plt.ylabel("Fraction of input in explanation")
plt.savefig("graphs/exp_size.pdf", format='pdf')

# Runtime of average runtime of Gurobi vs size
plt.figure()
for domain in domains.keys():    
    gurobi_min = [p[0] for p in gurobi_runtimes[domain]]
    gurobi_max = [p[1] for p in gurobi_runtimes[domain]]
    gurobi_mean = [p[2] for p in gurobi_runtimes[domain]]
    
    # Ensure the problems are in order
    gurobi_min = [x for _,x in sorted(zip(size[domain],gurobi_min))]
    gurobi_max = [x for _,x in sorted(zip(size[domain],gurobi_max))]
    gurobi_mean = [x for _,x in sorted(zip(size[domain],gurobi_mean))]
    size[domain] = sorted(size[domain])
    
    plt.plot(size[domain], gurobi_mean, label=domain, linestyle='--')
    plt.fill_between(size[domain], gurobi_min, gurobi_max, alpha=0.2)
    
plt.legend()
#plt.xlabel("Number of modules in the neural network")
plt.xlabel("Nb modules in neural network")
#plt.ylabel("Average runtime of MIP solver (seconds)")
plt.ylabel("Av. runtime of MIP solver (sec)")
plt.yscale('log')
plt.savefig("graphs/milp_runtime.pdf", format='pdf')

# Box plot of the avg number of calls / plan length in each domain
plt.figure()
# for domain in domains:
#     print(domain)
#     print(calls_over_length[domain])
plt.boxplot([calls_over_length[domain] for domain in domains], labels=domains.keys())

plt.legend()
plt.xlabel("Domains")
#plt.ylabel("Average MIP calls per proposition per plan step")
plt.ylabel("Av. MIP calls per proposition & plan step")
plt.savefig("graphs/avg_milp_calls.pdf", format='pdf')
# plt.show()
