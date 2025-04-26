import random
import numpy as np
import math

"""
5 component of PCID
Task_Instruction
Demonstration_Input
Demonstration_Label
Demonstration_Inline_Instruction
Test_Instance_Input
"""




def random_baseline():
    pmf = []
    sum = 0
    for i in range(5):
        mass = random.random()
        pmf.append(mass)
        sum += mass
    
    pmf = [mass / sum for mass in pmf]
    assert np.isclose(np.sum(pmf), 1), "The distribution doesn't sum to one"
    return pmf

def kl_divergence(P, Q):
    kl = 0.0
    for p, q in zip(P, Q):
        if p > 0:
            kl += p * math.log(p / q)
    return kl

def js_divergence(P, Q):
    M = [(p + q) / 2.0 for p, q in zip(P, Q)]
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)


def average_js_similarity(num_trials=10000):
    total_js = 0.0
    for _ in range(num_trials):
        pmf1 = random_baseline()
        pmf2 = random_baseline()
        total_js += js_divergence(pmf1, pmf2)
    return total_js / num_trials

def main():
    avg_js = average_js_similarity(num_trials=10000)
    print("Average Jensen-Shannon Divergence over 10,000 random PMF pairs:", avg_js)

if __name__ == "__main__":
    main()