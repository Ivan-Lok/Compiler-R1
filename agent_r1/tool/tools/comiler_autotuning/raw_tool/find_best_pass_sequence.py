#!/usr/bin/env python3
"""
Tool for finding the best LLVM optimization pass sequence for a given LLVM IR file.
This tool generates multiple pass sequences, applies them to the input file, and returns
the sequence that produces the lowest instruction count. (Single-Threaded Version)
"""

import os
import sys
import json
import time
# Removed concurrent.futures import
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor

# Import the required functions from existing modules
# Ensure these imports point to the correct location of your modules
try:
    from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_goodpasssequence import build_graph, generate_population, synerpairs
    from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import get_instrcount
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure the script is run from a location where 'agent_r1' package is accessible,")
    print("or adjust the import paths accordingly.")
    sys.exit(1)

def read_llvm_ir_file(file_path: str) -> str:
    """Read the LLVM IR content from a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)


def LeverageSyner_GA(edges, ll_code, population, llvm_tools_path):
    import random

    Ori = get_instrcount(ll_code, [], llvm_tools_path=llvm_tools_path)

    # Create the graph
    graph = defaultdict(list)
    nodes = set()
    for start, end in edges:
        graph[start].append(end)
        nodes.add(start)
        nodes.add(end)

    # Genetic algorithm parameters
    GENERATIONS = 10
    MUTATION_RATE = 0.5
    SELECTION_RATE = 0.1
    POPULATION = population

    # Fitness function
    def fitness_function(path):
        score = Ori - get_instrcount(ll_code, path, llvm_tools_path=llvm_tools_path)
        return score, path

    def calculate_fitness(population):
        fitness_scores = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(fitness_function, path) for path in population]
            for future in futures:
                max_score, best_sub_path = future.result()  # Get the maximum score and corresponding sub-path
                fitness_scores.append((max_score, best_sub_path))  # Store the maximum score and sub-path
        return sorted(fitness_scores, key=lambda x: x[0], reverse=True)

    # Selection
    def selection(fitness_scores, rate):
        selected = fitness_scores[:int(len(fitness_scores) * rate)]
        return [path for _, path in selected]

    # Crossover
    def crossover(parent1, parent2):
        common_nodes = set(parent1) & set(parent2)
        if not common_nodes:
            return parent1, parent2

        crossover_node = random.choice(list(common_nodes))
        idx1 = parent1.index(crossover_node)
        idx2 = parent2.index(crossover_node)

        child1 = parent1[:idx1] + parent2[idx2:]
        child2 = parent2[:idx2] + parent1[idx1:]

        return child1, child2

    def find_parents_with_common_nodes(selected):
        attempts = 0
        while attempts < 10:
            parent1, parent2 = random.sample(selected, 2)
            if set(parent1) & set(parent2):
                return parent1, parent2
            attempts += 1
        return selected[0], selected[1]

    # Mutation
    def mutate(path, mutation_rate):
        if random.random() < mutation_rate:
            mutation_points = [i for i, node in enumerate(path) if len(graph[node]) > 1]
            if mutation_points:
                mutation_point = random.choice(mutation_points)
                current = path[mutation_point]
                next_node = random.choice(graph[current])
                mutated_path = path[:mutation_point + 1]
                mutated_path.append(next_node)
                current = next_node
                
                while current in graph and graph[current]:
                    next_nodes = graph[current]
                    next_node = random.choice(next_nodes)
                    if next_node not in mutated_path:
                        mutated_path.append(next_node)
                        current = next_node
                    else:
                        break
                
                path = mutated_path
        return path

    # Main genetic algorithm function
    def genetic_algorithm(generations, mutation_rate, selection_rate, population):
        population_size = len(population)
        for i in range(generations):
            fitness_scores = calculate_fitness(population)
            selected = selection(fitness_scores, selection_rate)
            next_population = []
            while len(next_population) < population_size:
                parent1, parent2 = find_parents_with_common_nodes(selected)
                child1, child2 = crossover(parent1, parent2)
                next_population.append(mutate(child1, mutation_rate))
                next_population.append(mutate(child2, mutation_rate))
            population = next_population
        final_fitness_scores = calculate_fitness(population)
        best_path = final_fitness_scores[0][1]
        best_cost = final_fitness_scores[0][0]
        return best_path, best_cost
    
    best_path, best_cost = genetic_algorithm(GENERATIONS, MUTATION_RATE, SELECTION_RATE, POPULATION)
    # print("Best path: ", best_path)
    # print("Best Score: ", best_cost)
    # return best_cost, best_path
    Ox = Ori - best_cost

    return Ox, best_path

def find_best_pass_sequence(file_path: str, llvm_tools_path: str,
                            population_size: int = 100,
                            max_length: int = 30,
                            min_length: int = 10) -> Dict: # Removed max_workers
    """
    Find the best pass sequence for the given LLVM IR file using single-threading.

    Args:
        file_path: Path to the LLVM IR file
        llvm_tools_path: Path to LLVM tools
        population_size: Number of pass sequences to generate
        max_length: Maximum length of a pass sequence
        min_length: Minimum length of a pass sequence

    Returns:
        Dictionary containing the best pass sequence and its improvement percentage.
    """
    # Removed max_workers calculation

    # Read the LLVM IR code from the file
    ll_code = read_llvm_ir_file(file_path)

    # Generate a population of pass sequences
    pass_sequences = generate_population(synerpairs, size=population_size,
                                         max_length=max_length, min_length=min_length)

    # Removed task list creation

    # Track best sequence and its instruction count
    best_sequence = None
    best_instr_count = float('inf')

    # Removed progress tracking setup

    # --- Start Sequential Evaluation ---
    best_instr_count, best_sequence = LeverageSyner_GA(synerpairs, ll_code, pass_sequences, llvm_tools_path)


    # Evaluate the baseline '-Oz' sequence
    baseline_instr_count = get_instrcount(ll_code, ['-Oz'], llvm_tools_path=llvm_tools_path)

    # Handle cases where evaluation might fail
    if baseline_instr_count is None:
        # If baseline fails, improvement cannot be calculated meaningfully
        # Return best found sequence, or default to Oz if none worked.
        improvement = 0 # Or signal error? Keep simple for now.
        if best_sequence is None:
            best_sequence = ['-Oz'] # Default if nothing worked
    elif best_instr_count == float('inf'):
        # No generated sequence was successfully evaluated or improved
        improvement = 0
        best_sequence = ['-Oz'] # Default to Oz
    elif baseline_instr_count == 0:
         # Avoid division by zero
         improvement = 0
         if best_sequence is None: # Ensure best_sequence is not None
             best_sequence = ['-Oz']
    else:
        # Calculate improvement percentage
        improvement = ((baseline_instr_count - best_instr_count) / baseline_instr_count) * 100
        # If best_sequence is still None here (only possible if all evaluated sequences failed
        # but baseline succeeded), default to Oz.
        if best_sequence is None:
             best_sequence = ['-Oz']
             improvement = 0 # Revert improvement if we fallback


    result = {
        "best_pass_sequence": best_sequence,
        "improvement_percentage": round(improvement, 2),
    }

    return result
