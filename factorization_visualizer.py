#!/usr/bin/env python
"""
Advanced Prime Factorization Visualizer

This module creates interactive, visually rich visualizations for prime factorization
decomposition trees and algorithm execution traces, helping to understand the
factorization process.
"""

import math
import random
import sympy
import time
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Optional imports for enhanced visualization
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class FactorizationVisualizer:
    """Base class for prime factorization visualizers."""
    
    def __init__(self, dark_mode: bool = False, output_dir: str = "visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            dark_mode: Whether to use dark mode for visualizations
            output_dir: Directory to save visualizations
        """
        self.dark_mode = dark_mode
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set up styling based on dark_mode
        if dark_mode:
            plt.style.use('dark_background')
            self.bg_color = '#121212'
            self.text_color = 'white'
            self.edge_color = '#666666'
            self.node_colors = ['#1976D2', '#E53935', '#43A047', '#FFC107', '#5E35B1']
            self.highlight_color = '#FF5722'
        else:
            plt.style.use('default')
            self.bg_color = 'white'
            self.text_color = 'black'
            self.edge_color = '#333333'
            self.node_colors = ['#2196F3', '#F44336', '#4CAF50', '#FFC107', '#673AB7']
            self.highlight_color = '#FF5722'
    
    def visualize_factorization(self, number: int, factors: List[int], 
                              filename: str = None) -> str:
        """
        Create a basic visualization of prime factorization.
        
        Args:
            number: The number being factorized
            factors: List of prime factors
            filename: Optional filename to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        # Use default filename if not provided
        if filename is None:
            filename = f"factorization_{number}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor(self.bg_color)
        fig.set_facecolor(self.bg_color)
        
        # Draw prime factorization as a simple equation
        factors_str = ' × '.join(map(str, factors))
        equation = f"{number} = {factors_str}"
        
        ax.text(0.5, 0.5, equation, fontsize=18, ha='center', va='center', color=self.text_color)
        
        # Remove axis
        ax.axis('off')
        
        # Set title
        ax.set_title(f"Prime Factorization of {number}", color=self.text_color, fontsize=20)
        
        # Save and return
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
    
    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime."""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True


class DecompositionTreeVisualizer(FactorizationVisualizer):
    """Visualizes factorization as a decomposition tree."""
    
    def visualize_decomposition_tree(self, number: int, factors: List[int] = None, 
                                    show_steps: bool = True,
                                    filename: str = None) -> str:
        """
        Create a visualization of prime factorization as a decomposition tree.
        
        Args:
            number: The number being factorized
            factors: Optional list of prime factors (if not provided, will be calculated)
            show_steps: Whether to show the step-by-step decomposition
            filename: Optional filename to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not HAS_NETWORKX:
            return self._fallback_tree_visualization(number, factors, filename)
        
        # Calculate factors if not provided
        if factors is None:
            factors = self.factorize(number)
        
        # Use default filename if not provided
        if filename is None:
            filename = f"decomposition_tree_{number}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Build the decomposition tree
        self._build_tree(G, number, factors)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor(self.bg_color)
        fig.set_facecolor(self.bg_color)
        
        # Set up layout
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot") if hasattr(nx, 'nx_agraph') else nx.spring_layout(G, seed=42)
        
        # Draw nodes with different colors based on level
        levels = {}
        for node in G.nodes():
            # Identify the level based on distance from root
            if node == str(number):
                levels[node] = 0
            else:
                # Find parent
                for parent, child in G.edges():
                    if child == node:
                        levels[node] = levels.get(parent, 0) + 1
                        break
        
        # Draw nodes by level
        max_level = max(levels.values()) if levels else 0
        for level in range(max_level + 1):
            nodes_at_level = [node for node, node_level in levels.items() if node_level == level]
            
            # Determine color for this level
            color_idx = level % len(self.node_colors)
            color = self.node_colors[color_idx]
            
            # Determine node size based on level
            size = 1000 * (0.8 ** level)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_at_level, 
                                 node_color=color, node_size=size, alpha=0.8, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color=self.edge_color, width=1.5, 
                              alpha=0.7, arrows=True, arrowsize=15, ax=ax)
        
        # Draw labels
        node_labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, 
                              font_color=self.text_color, ax=ax)
        
        # Add prime indicators
        for node in G.nodes():
            node_val = int(node)
            if self._is_prime(node_val):
                # Find position and add a star or indicator
                node_pos = pos[node]
                ax.plot(node_pos[0], node_pos[1], marker='*', markersize=15, 
                       color=self.highlight_color, alpha=0.9)
        
        # Add legend
        legend_elements = [
            patches.Patch(color=self.node_colors[0], label='Original Number'),
            patches.Patch(color=self.node_colors[1], label='First Level Factors'),
            patches.Patch(color=self.node_colors[2], label='Second Level Factors'),
            patches.Patch(color=self.highlight_color, label='Prime Numbers')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add title and remove axis
        ax.set_title(f"Prime Factorization Tree for {number}", fontsize=16, color=self.text_color)
        ax.axis('off')
        
        # Add factorization equation at the bottom
        factors_str = ' × '.join(map(str, factors))
        equation = f"{number} = {factors_str}"
        fig.text(0.5, 0.01, equation, fontsize=14, ha='center', color=self.text_color)
        
        # Save and return
        plt.tight_layout()
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        # If show_steps is True, also create an animated step-by-step visualization
        if show_steps:
            step_filename = f"decomposition_steps_{number}.gif"
            step_filepath = os.path.join(self.output_dir, step_filename)
            self._create_step_animation(number, factors, step_filepath)
            return step_filepath
        
        return filepath
    
    def _build_tree(self, G, number, all_factors, parent=None):
        """Recursively build the factorization tree."""
        number_str = str(number)
        
        # Add the node if it doesn't exist
        if number_str not in G.nodes():
            G.add_node(number_str)
        
        # Add edge from parent if this is not the root
        if parent is not None:
            G.add_edge(parent, number_str)
        
        # If the number is prime, we're done with this branch
        if self._is_prime(number):
            return
        
        # Find the smallest factor
        smallest_factor = None
        for factor in all_factors:
            if number % factor == 0:
                smallest_factor = factor
                break
        
        # If no factor found (shouldn't happen if all_factors is complete), stop
        if smallest_factor is None:
            return
        
        # Calculate the cofactor
        cofactor = number // smallest_factor
        
        # Add the factor and cofactor to the graph
        factor_str = str(smallest_factor)
        cofactor_str = str(cofactor)
        
        # Continue building the tree
        self._build_tree(G, smallest_factor, all_factors, number_str)
        self._build_tree(G, cofactor, all_factors, number_str)
    
    def _create_step_animation(self, number, factors, filepath):
        """Create a step-by-step animation of the factorization process."""
        if not HAS_NETWORKX:
            return None
        
        # Create a series of graphs representing each step
        steps = []
        
        # Track the factorization process
        remaining = number
        step_factors = []
        
        # Create the initial graph with just the number
        G = nx.DiGraph()
        G.add_node(str(number))
        steps.append(G.copy())
        
        # Build up the factorization step by step
        for factor in factors:
            while remaining % factor == 0:
                # Create a new graph
                G = steps[-1].copy()
                
                # Find the leaf nodes (remaining composites)
                leaf_nodes = [node for node in G.nodes() if G.out_degree(node) == 0 
                           and not self._is_prime(int(node)) and int(node) > 1]
                
                # Find a leaf node divisible by this factor
                for leaf in leaf_nodes:
                    leaf_val = int(leaf)
                    if leaf_val % factor == 0:
                        # Add factor and cofactor
                        cofactor = leaf_val // factor
                        G.add_node(str(factor))
                        G.add_node(str(cofactor))
                        G.add_edge(leaf, str(factor))
                        G.add_edge(leaf, str(cofactor))
                        
                        # Add this step
                        steps.append(G.copy())
                        
                        # Update remaining
                        remaining = max(remaining // factor, cofactor)
                        step_factors.append(factor)
                        break
        
        # Create the animation
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor(self.bg_color)
        fig.set_facecolor(self.bg_color)
        
        def update(frame):
            ax.clear()
            G = steps[frame]
            
            # Set up layout
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot") if hasattr(nx, 'nx_agraph') else nx.spring_layout(G, seed=42)
            
            # Draw nodes with different colors based on level
            levels = {}
            for node in G.nodes():
                # Identify the level based on distance from root
                if node == str(number):
                    levels[node] = 0
                else:
                    # Find parent using BFS
                    queue = [str(number)]
                    visited = set(queue)
                    current_level = 0
                    
                    while queue:
                        current_level += 1
                        next_queue = []
                        
                        for n in queue:
                            for child in G.successors(n):
                                if child == node:
                                    levels[node] = current_level
                                if child not in visited:
                                    visited.add(child)
                                    next_queue.append(child)
                        
                        queue = next_queue
            
            # Draw nodes by level
            max_level = max(levels.values()) if levels else 0
            for level in range(max_level + 1):
                nodes_at_level = [node for node, node_level in levels.items() if node_level == level]
                
                # Determine color for this level
                color_idx = level % len(self.node_colors)
                color = self.node_colors[color_idx]
                
                # Determine node size based on level
                size = 1000 * (0.8 ** level)
                
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, nodelist=nodes_at_level, 
                                    node_color=color, node_size=size, alpha=0.8, ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color=self.edge_color, width=1.5, 
                                alpha=0.7, arrows=True, arrowsize=15, ax=ax)
            
            # Draw labels
            node_labels = {node: node for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, 
                                font_color=self.text_color, ax=ax)
            
            # Add prime indicators
            for node in G.nodes():
                node_val = int(node)
                if self._is_prime(node_val):
                    # Find position and add a star or indicator
                    node_pos = pos[node]
                    ax.plot(node_pos[0], node_pos[1], marker='*', markersize=15, 
                           color=self.highlight_color, alpha=0.9)
            
            # Add title
            factors_so_far = step_factors[:frame] if frame > 0 else []
            if factors_so_far:
                factors_str = ' × '.join(map(str, factors_so_far))
                subtitle = f"Factors found: {factors_str}"
            else:
                subtitle = "Initial state"
            
            ax.set_title(f"Prime Factorization of {number}: Step {frame+1}/{len(steps)}\n{subtitle}", 
                       fontsize=14, color=self.text_color)
            
            ax.axis('off')
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=1000)
        
        # Save animation
        ani.save(filepath, writer='pillow', fps=1, dpi=100)
        plt.close(fig)
        
        return filepath
    
    def _fallback_tree_visualization(self, number, factors, filename):
        """Create a simpler tree visualization without NetworkX."""
        # Calculate factors if not provided
        if factors is None:
            factors = self.factorize(number)
        
        # Use default filename if not provided
        if filename is None:
            filename = f"decomposition_tree_{number}_simple.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor(self.bg_color)
        fig.set_facecolor(self.bg_color)
        
        # Create a simple tree visualization
        factor_tree = self._create_factor_tree(number, factors)
        
        def plot_tree(node, x, y, dx, level=0):
            # Plot the current node
            color_idx = level % len(self.node_colors)
            color = self.node_colors[color_idx]
            
            # Draw the node
            circle = plt.Circle((x, y), 0.1, color=color)
            ax.add_patch(circle)
            
            # Add text for the value
            ax.text(x, y, str(node['value']), ha='center', va='center', 
                  color=self.text_color, fontweight='bold')
            
            # If this is a prime, add a star
            if node.get('is_prime', False):
                ax.plot(x, y + 0.15, marker='*', markersize=15, 
                       color=self.highlight_color)
            
            # Plot children and connect them to the parent
            if 'children' in node:
                n_children = len(node['children'])
                child_dx = dx / max(1, n_children)
                
                for i, child in enumerate(node['children']):
                    child_x = x - dx/2 + child_dx/2 + i * child_dx
                    child_y = y - 0.5
                    
                    # Draw edge
                    ax.plot([x, child_x], [y - 0.1, child_y + 0.1], 
                           color=self.edge_color, linestyle='-', linewidth=1)
                    
                    # Recursively plot the child
                    plot_tree(child, child_x, child_y, child_dx/2, level + 1)
        
        # Plot the tree
        plot_tree(factor_tree, 0.5, 0.9, 0.4)
        
        # Set limits and remove ticks
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title
        ax.set_title(f"Prime Factorization Tree for {number}", fontsize=16, color=self.text_color)
        
        # Add factorization equation at the bottom
        factors_str = ' × '.join(map(str, factors))
        equation = f"{number} = {factors_str}"
        fig.text(0.5, 0.05, equation, fontsize=14, ha='center', color=self.text_color)
        
        # Save and return
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
    
    def _create_factor_tree(self, number, all_factors):
        """Create a factor tree data structure."""
        # Base case: if the number is prime
        if self._is_prime(number):
            return {'value': number, 'is_prime': True}
        
        # Find a factor
        factor = None
        for f in all_factors:
            if number % f == 0:
                factor = f
                break
        
        if factor is None:
            return {'value': number, 'is_prime': True}  # Fallback to treating as prime
        
        # Get the cofactor
        cofactor = number // factor
        
        # Create node for current number
        node = {'value': number, 'is_prime': False}
        
        # Create child nodes
        node['children'] = [
            self._create_factor_tree(factor, all_factors),
            self._create_factor_tree(cofactor, all_factors)
        ]
        
        return node
    
    def factorize(self, n):
        """Simple factorization function for demonstration."""
        if n <= 1:
            return [n]
        
        factors = []
        d = 2
        
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1 if d == 2 else 2
        
        if n > 1:
            factors.append(n)
        
        return factors


class AlgorithmVisualizer(FactorizationVisualizer):
    """Visualizes the execution of factorization algorithms."""
    
    def visualize_algorithm(self, algorithm_name: str, number: int, 
                           execution_trace: List[Dict] = None,
                           filename: str = None) -> str:
        """
        Create a visualization of algorithm execution.
        
        Args:
            algorithm_name: Name of the factorization algorithm
            number: The number being factorized
            execution_trace: Optional trace of algorithm execution steps
            filename: Optional filename to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        # Generate a trace if not provided
        if execution_trace is None:
            execution_trace = self._generate_sample_trace(algorithm_name, number)
        
        # Use default filename if not provided
        if filename is None:
            filename = f"{algorithm_name}_visualization_{number}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Select the appropriate visualization method based on algorithm
        if algorithm_name.lower() == "trial_division":
            return self._visualize_trial_division(number, execution_trace, filepath)
        elif algorithm_name.lower() == "pollard_rho":
            return self._visualize_pollard_rho(number, execution_trace, filepath)
        elif algorithm_name.lower() == "quadratic_sieve":
            return self._visualize_quadratic_sieve(number, execution_trace, filepath)
        else:
            # Generic visualization for other algorithms
            return self._visualize_generic_algorithm(algorithm_name, number, execution_trace, filepath)
    
    def _generate_sample_trace(self, algorithm_name, number):
        """Generate a sample execution trace for demonstration."""
        if algorithm_name.lower() == "trial_division":
            return self._generate_trial_division_trace(number)
        elif algorithm_name.lower() == "pollard_rho":
            return self._generate_pollard_rho_trace(number)
        elif algorithm_name.lower() == "quadratic_sieve":
            return self._generate_quadratic_sieve_trace(number)
        else:
            # Generic trace
            return [{"step": i, "description": f"Step {i}", "value": None} for i in range(5)]
    
    def _generate_trial_division_trace(self, number):
        """Generate a trace for trial division algorithm."""
        trace = []
        n = number
        divisor = 2
        step = 1
        
        while divisor * divisor <= n:
            while n % divisor == 0:
                trace.append({
                    "step": step,
                    "divisor": divisor,
                    "number": n,
                    "is_factor": True,
                    "result": n // divisor,
                    "description": f"Found factor: {divisor}"
                })
                n //= divisor
                step += 1
            
            # Also record failed attempts for visualization
            if n % divisor != 0:
                trace.append({
                    "step": step,
                    "divisor": divisor,
                    "number": n,
                    "is_factor": False,
                    "result": n,
                    "description": f"{divisor} is not a factor"
                })
                step += 1
            
            divisor += 1 if divisor == 2 else 2
        
        if n > 1:
            trace.append({
                "step": step,
                "divisor": n,
                "number": n,
                "is_factor": True,
                "result": 1,
                "description": f"Final prime factor: {n}"
            })
        
        return trace
    
    def _generate_pollard_rho_trace(self, number):
        """Generate a trace for Pollard's rho algorithm."""
        trace = []
        
        # For demonstration, generate a simplified trace
        x, y = 2, 2
        d = 1
        c = 1  # Constant for the polynomial
        step = 1
        n = number
        
        # Check if n is prime first
        if self._is_prime(n):
            trace.append({
                "step": step,
                "x": None,
                "y": None,
                "gcd": None,
                "description": f"{n} is prime"
            })
            return trace
        
        def g(x):
            return (x * x + c) % n
        
        while d == 1 and step <= 20:  # Limit steps for demonstration
            x = g(x)
            y = g(g(y))
            d = math.gcd(abs(x - y), n)
            
            trace.append({
                "step": step,
                "x": x,
                "y": y,
                "gcd": d,
                "description": f"x={x}, y={y}, gcd={d}"
            })
            
            step += 1
            
            # If we find a factor, add one more step showing the factorization
            if d > 1 and d < n:
                trace.append({
                    "step": step,
                    "x": None,
                    "y": None,
                    "gcd": d,
                    "factor": d,
                    "cofactor": n // d,
                    "description": f"Found factor {d}, cofactor {n // d}"
                })
                break
        
        # If no factor found after max iterations
        if d == 1 or d == n:
            trace.append({
                "step": step,
                "x": None,
                "y": None,
                "gcd": d,
                "description": "No factor found, would retry with different parameters"
            })
        
        return trace
    
    def _generate_quadratic_sieve_trace(self, number):
        """Generate a trace for quadratic sieve algorithm."""
        # This is a simplified trace for demonstration
        trace = []
        n = number
        step = 1
        
        # Start with the base range
        sqrt_n = int(math.sqrt(n))
        a = sqrt_n
        
        trace.append({
            "step": step,
            "description": f"Start with sqrt({n}) ≈ {sqrt_n}",
            "a": a,
            "b2": None,
            "smooth": False
        })
        
        step += 1
        
        # Try some values in the sieve
        for i in range(1, 6):  # Just a few steps for demonstration
            a = sqrt_n + i
            b2 = a * a - n
            
            is_smooth = False
            factors = []
            
            # Try to factor b2 with small primes for demonstration
            if b2 > 0:
                is_smooth = self._is_smooth(b2, 100)  # Check if smooth over small primes
                if is_smooth:
                    factors = self._trial_factorization(b2, 100)
            
            trace.append({
                "step": step,
                "description": f"Try a={a}, a²-n={b2}",
                "a": a,
                "b2": b2,
                "smooth": is_smooth,
                "factors": factors if is_smooth else None
            })
            
            step += 1
            
            # If we find a smooth value, pretend we've made progress
            if is_smooth:
                trace.append({
                    "step": step,
                    "description": f"Found smooth value: {b2} = {' × '.join(map(str, factors))}",
                    "a": a,
                    "b2": b2,
                    "smooth": True,
                    "factors": factors
                })
                step += 1
                break
        
        # Add a final step summarizing the result
        if n % 2 == 0:  # Simple case for demonstration
            trace.append({
                "step": step,
                "description": f"Found factorization: {n} = 2 × {n//2}",
                "factors": [2, n//2]
            })
        else:
            # Random factor for demonstration
            potential_factors = [3, 5, 7, 11, 13]
            for f in potential_factors:
                if n % f == 0:
                    trace.append({
                        "step": step,
                        "description": f"Found factorization: {n} = {f} × {n//f}",
                        "factors": [f, n//f]
                    })
                    break
            else:
                trace.append({
                    "step": step,
                    "description": f"Full factorization would require more steps"
                })
        
        return trace
    
    def _is_smooth(self, n, bound):
        """Check if a number is B-smooth (all prime factors <= bound)."""
        if n <= 1:
            return True
        
        for p in range(2, bound + 1):
            if not self._is_prime(p):
                continue
            
            while n % p == 0:
                n //= p
            
            if n == 1:
                return True
        
        return n == 1
    
    def _trial_factorization(self, n, bound):
        """Simple factorization up to a bound."""
        factors = []
        for p in range(2, bound + 1):
            if not self._is_prime(p):
                continue
            
            while n % p == 0:
                factors.append(p)
                n //= p
        
        if n > 1:
            factors.append(n)
        
        return factors
    
    def _visualize_trial_division(self, number, trace, filepath):
        """Visualize trial division algorithm."""
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.patch.set_facecolor(self.bg_color)
        ax1.set_facecolor(self.bg_color)
        ax2.set_facecolor(self.bg_color)
        
        # Extract data for visualization
        steps = [item["step"] for item in trace]
        divisors = [item["divisor"] for item in trace]
        remaining = [item["number"] for item in trace]
        is_factor = [item["is_factor"] for item in trace]
        
        # Plot divisors tested
        colors = [self.highlight_color if factor else self.edge_color 
                for factor in is_factor]
        
        ax1.bar(steps, divisors, color=colors)
        ax1.set_xlabel("Step", color=self.text_color)
        ax1.set_ylabel("Divisor", color=self.text_color)
        ax1.set_title("Divisors Tested", color=self.text_color)
        ax1.tick_params(colors=self.text_color)
        
        # Plot remaining number
        ax2.plot(steps, remaining, marker='o', linestyle='-', color=self.node_colors[0])
        ax2.set_xlabel("Step", color=self.text_color)
        ax2.set_ylabel("Remaining Number", color=self.text_color)
        ax2.set_title("Factorization Progress", color=self.text_color)
        ax2.tick_params(colors=self.text_color)
        
        # Add main title
        fig.suptitle(f"Trial Division Algorithm: Factorizing {number}", fontsize=16, 
                   color=self.text_color)
        
        # Add legend for factor/non-factor
        import matplotlib.lines as mlines
        factor_legend = mlines.Line2D([], [], color=self.highlight_color, marker='s', 
                                    linestyle='None', markersize=10, label='Factor')
        non_factor_legend = mlines.Line2D([], [], color=self.edge_color, marker='s', 
                                       linestyle='None', markersize=10, label='Not a Factor')
        ax1.legend(handles=[factor_legend, non_factor_legend])
        
        # Add description of each step
        description_text = ""
        for i, item in enumerate(trace):
            if i < 10:  # Limit to first 10 steps for clarity
                description_text += f"Step {item['step']}: {item.get('description', '')}\n"
        
        # Add complete factorization at the bottom
        factors = [item["divisor"] for item in trace if item["is_factor"]]
        if factors:
            factors_str = ' × '.join(map(str, factors))
            description_text += f"\nComplete factorization: {number} = {factors_str}"
        
        fig.text(0.5, 0.01, description_text, fontsize=10, ha='center', 
               color=self.text_color)
        
        # Save the visualization
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
    
    def _visualize_pollard_rho(self, number, trace, filepath):
        """Visualize Pollard's rho algorithm."""
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.patch.set_facecolor(self.bg_color)
        ax1.set_facecolor(self.bg_color)
        ax2.set_facecolor(self.bg_color)
        
        # Extract data for visualization
        steps = []
        x_values = []
        y_values = []
        gcd_values = []
        
        for item in trace:
            # Only plot steps with valid x, y values
            if "x" in item and "y" in item and item["x"] is not None and item["y"] is not None:
                steps.append(item["step"])
                x_values.append(item["x"])
                y_values.append(item["y"])
                gcd_values.append(item.get("gcd", 1))
        
        # Plot x and y values
        if x_values and y_values:
            ax1.plot(steps, x_values, marker='o', linestyle='-', color=self.node_colors[0], label='x')
            ax1.plot(steps, y_values, marker='s', linestyle='-', color=self.node_colors[1], label='y')
            ax1.set_xlabel("Step", color=self.text_color)
            ax1.set_ylabel("Value", color=self.text_color)
            ax1.set_title("Sequence Values", color=self.text_color)
            ax1.tick_params(colors=self.text_color)
            ax1.legend()
            
            # Plot modular differences on a circle
            if len(steps) > 0:
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = np.cos(theta)
                circle_y = np.sin(theta)
                ax2.plot(circle_x, circle_y, color=self.edge_color, alpha=0.3)
                
                # Plot points on the circle
                for i in range(len(steps)):
                    angle1 = 2 * np.pi * (x_values[i] / number)
                    angle2 = 2 * np.pi * (y_values[i] / number)
                    ax2.plot(np.cos(angle1), np.sin(angle1), 'o', color=self.node_colors[0])
                    ax2.plot(np.cos(angle2), np.sin(angle2), 's', color=self.node_colors[1])
                    
                    # Draw a chord connecting x and y
                    ax2.plot([np.cos(angle1), np.cos(angle2)], 
                           [np.sin(angle1), np.sin(angle2)], 
                           linestyle='--', color=self.edge_color, alpha=0.5)
                
                ax2.set_aspect('equal')
                ax2.axis('off')
                ax2.set_title("Modular Representation", color=self.text_color)
        
        # Add main title
        fig.suptitle(f"Pollard's Rho Algorithm: Factorizing {number}", fontsize=16, 
                   color=self.text_color)
        
        # Add description of each step
        description_text = "Algorithm Steps:\n"
        for i, item in enumerate(trace):
            if i < 10:  # Limit to first 10 steps for clarity
                description_text += f"Step {item['step']}: {item.get('description', '')}\n"
        
        # Add factorization result at the bottom if found
        found_factors = False
        for item in trace:
            if "factor" in item:
                description_text += f"\nFound factorization: {number} = {item['factor']} × {item['cofactor']}"
                found_factors = True
                break
        
        if not found_factors:
            description_text += f"\nComplete factorization would require more steps or different parameters."
        
        fig.text(0.5, 0.01, description_text, fontsize=10, ha='center', 
               color=self.text_color)
        
        # Save the visualization
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
    
    def _visualize_quadratic_sieve(self, number, trace, filepath):
        """Visualize Quadratic Sieve algorithm."""
        if HAS_PLOTLY:
            return self._plotly_qs_visualization(number, trace, filepath)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.patch.set_facecolor(self.bg_color)
        ax1.set_facecolor(self.bg_color)
        ax2.set_facecolor(self.bg_color)
        
        # Extract data for visualization
        steps = []
        a_values = []
        b2_values = []
        smooth = []
        
        for item in trace:
            if "a" in item and item["a"] is not None:
                steps.append(item["step"])
                a_values.append(item["a"])
                b2 = item.get("b2", None)
                b2_values.append(b2 if b2 is not None else None)
                smooth.append(item.get("smooth", False))
        
        # Plot a values and a²-n
        ax1.plot(steps, a_values, marker='o', linestyle='-', color=self.node_colors[0], label='a')
        
        valid_points = [(s, b2) for s, b2 in zip(steps, b2_values) if b2 is not None]
        if valid_points:
            valid_steps, valid_b2 = zip(*valid_points)
            ax1.plot(valid_steps, valid_b2, marker='s', linestyle='-', color=self.node_colors[1], label='a²-n')
        
        ax1.set_xlabel("Step", color=self.text_color)
        ax1.set_ylabel("Value", color=self.text_color)
        ax1.set_title("Quadratic Sieve Values", color=self.text_color)
        ax1.tick_params(colors=self.text_color)
        ax1.legend()
        
        # Visualize the sieving process in 2D
        sqrt_n = int(math.sqrt(number))
        x = np.linspace(sqrt_n - 5, sqrt_n + 5, 100)
        y = x**2 - number
        
        ax2.plot(x, y, color=self.edge_color)
        ax2.axhline(y=0, color=self.text_color, linestyle='--', alpha=0.5)
        
        # Plot the tested points
        for i, (a, b2, is_smooth) in enumerate(zip(a_values, b2_values, smooth)):
            if b2 is not None:
                color = self.highlight_color if is_smooth else self.edge_color
                ax2.plot(a, b2, 'o', color=color, markersize=10)
                ax2.text(a, b2, f"{i+1}", color=self.text_color, fontsize=8)
        
        ax2.set_xlabel("a", color=self.text_color)
        ax2.set_ylabel("a²-n", color=self.text_color)
        ax2.set_title("Polynomial Values", color=self.text_color)
        ax2.tick_params(colors=self.text_color)
        
        # Add main title
        fig.suptitle(f"Quadratic Sieve Algorithm: Factorizing {number}", fontsize=16, 
                   color=self.text_color)
        
        # Add description of each step
        description_text = "Algorithm Steps:\n"
        for i, item in enumerate(trace):
            if i < 7:  # Limit to first 7 steps for clarity
                description_text += f"Step {item['step']}: {item.get('description', '')}\n"
        
        # Add factorization result at the bottom
        for item in trace:
            if "factors" in item and item["factors"]:
                description_text += f"\nFactors found: {' × '.join(map(str, item['factors']))}"
                break
        
        fig.text(0.5, 0.01, description_text, fontsize=10, ha='center', 
               color=self.text_color)
        
        # Save the visualization
        plt.tight_layout(rect=[0, 0.2, 1, 0.95])
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
    
    def _plotly_qs_visualization(self, number, trace, filepath):
        """Create an interactive Plotly visualization for Quadratic Sieve."""
        # Extract data
        steps = []
        a_values = []
        b2_values = []
        smooth_status = []
        descriptions = []
        
        for item in trace:
            if "a" in item and item["a"] is not None:
                steps.append(item["step"])
                a_values.append(item["a"])
                b2 = item.get("b2", None)
                b2_values.append(b2 if b2 is not None else None)
                smooth_status.append("Smooth" if item.get("smooth", False) else "Not Smooth")
                descriptions.append(item.get("description", ""))
        
        # Create the 2D polynomial visualization
        sqrt_n = int(math.sqrt(number))
        x = np.linspace(sqrt_n - 5, sqrt_n + 5, 100)
        y = x**2 - number
        
        # Create a plotly figure
        fig = go.Figure()
        
        # Add the polynomial curve
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=f'f(a) = a² - {number}',
            line=dict(color='gray')
        ))
        
        # Add the tested points
        valid_points = [(a, b2, status, desc) for a, b2, status, desc in 
                       zip(a_values, b2_values, smooth_status, descriptions) 
                       if b2 is not None]
        
        if valid_points:
            a_valid, b2_valid, status_valid, desc_valid = zip(*valid_points)
            
            colors = ['red' if status == 'Not Smooth' else 'green' for status in status_valid]
            
            fig.add_trace(go.Scatter(
                x=a_valid, y=b2_valid,
                mode='markers+text',
                name='Tested Values',
                marker=dict(color=colors, size=10),
                text=[f"{i+1}" for i in range(len(a_valid))],
                textposition="top center",
                hovertemplate='<b>a=%{x}</b><br>a²-n=%{y}<br>Status: %{text}<br>%{customdata}',
                customdata=desc_valid,
                textfont=dict(color='white')
            ))
        
        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=min(x), x1=max(x),
            y0=0, y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        # Update layout
        theme = 'plotly_dark' if self.dark_mode else 'plotly_white'
        
        fig.update_layout(
            title=f"Quadratic Sieve Algorithm: Factorizing {number}",
            template=theme,
            hovermode="closest",
            xaxis_title="a",
            yaxis_title="a²-n",
            legend_title="Legend",
            autosize=True,
            height=600
        )
        
        # Add a slider for steps
        steps_slider = []
        for i, (step, desc) in enumerate(zip(steps, descriptions)):
            visible = [False] * len(fig.data)
            visible[0] = True  # Always show the curve
            
            # Show points up to this step
            if i > 0 and 1 < len(visible):
                visible[1] = True
            
            step_data = {
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Step {step}: {desc}"}
                ],
                "label": f"Step {step}"
            }
            steps_slider.append(step_data)
        
        sliders = [dict(
            active=0,
            steps=steps_slider
        )]
        
        fig.update_layout(sliders=sliders)
        
        # Save as HTML and PNG
        html_path = filepath.replace('.png', '.html')
        fig.write_html(html_path)
        fig.write_image(filepath, width=1000, height=600)
        
        return html_path
    
    def _visualize_generic_algorithm(self, algorithm_name, number, trace, filepath):
        """Generic visualization for any algorithm."""
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)
        
        # Plot algorithm steps
        steps = [item["step"] for item in trace]
        values = []
        for item in trace:
            # Extract some numeric value to plot
            val = None
            for key in ["value", "result", "divisor", "gcd", "a"]:
                if key in item and item[key] is not None:
                    val = item[key]
                    break
            values.append(val)
        
        # Filter out None values
        valid_data = [(s, v) for s, v in zip(steps, values) if v is not None]
        if valid_data:
            valid_steps, valid_values = zip(*valid_data)
            ax.plot(valid_steps, valid_values, marker='o', linestyle='-', 
                  color=self.node_colors[0])
        
        ax.set_xlabel("Step", color=self.text_color)
        ax.set_ylabel("Value", color=self.text_color)
        ax.set_title(f"{algorithm_name} Progress", color=self.text_color)
        ax.tick_params(colors=self.text_color)
        
        # Add step descriptions
        description_text = "Algorithm Steps:\n"
        for i, item in enumerate(trace):
            if i < 15:  # Limit to first 15 steps for clarity
                description_text += f"Step {item['step']}: {item.get('description', '')}\n"
        
        fig.text(0.5, 0.01, description_text, fontsize=10, ha='center', 
               color=self.text_color)
        
        # Add main title
        fig.suptitle(f"{algorithm_name} Algorithm: Factorizing {number}", fontsize=16, 
                   color=self.text_color)
        
        # Save the visualization
        plt.tight_layout(rect=[0, 0.2, 1, 0.95])
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return filepath


class FactorizationTreeVisualizer(FactorizationVisualizer):
    """Visualizes factorization process in multiple dimensions."""
    
    def visualize_3d_factorization(self, number: int, factors: List[int] = None,
                                 filename: str = None) -> str:
        """
        Create a 3D visualization of the factorization process.
        
        Args:
            number: The number being factorized
            factors: Optional list of prime factors (if not provided, will be calculated)
            filename: Optional filename to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        # Calculate factors if not provided
        if factors is None:
            factors = self.factorize(number)
        
        # Use default filename if not provided
        if filename is None:
            filename = f"3d_factorization_{number}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create 3D figure
        fig = plt.figure(figsize=(12, 10))
        fig.patch.set_facecolor(self.bg_color)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(self.bg_color)
        
        # Generate coordinate data for the factorization
        x_coords, y_coords, z_coords, colors, sizes = self._generate_3d_coordinates(number, factors)
        
        # Plot the points
        scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, s=sizes, alpha=0.8)
        
        # Add connecting lines
        for i in range(1, len(x_coords)):
            ax.plot([x_coords[0], x_coords[i]], [y_coords[0], y_coords[i]], 
                  [z_coords[0], z_coords[i]], color=self.edge_color, alpha=0.5)
        
        # Connect prime factors in a circle
        prime_indices = [i for i in range(1, len(x_coords)) if self._is_prime(factors[i-1])]
        if len(prime_indices) > 1:
            for i in range(len(prime_indices)):
                next_i = (i + 1) % len(prime_indices)
                idx1, idx2 = prime_indices[i], prime_indices[next_i]
                ax.plot([x_coords[idx1], x_coords[idx2]], 
                      [y_coords[idx1], y_coords[idx2]],
                      [z_coords[idx1], z_coords[idx2]], 
                      color=self.highlight_color, alpha=0.5, linestyle='--')
        
        # Add labels to points
        for i, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)):
            label = str(number) if i == 0 else str(factors[i-1])
            ax.text(x, y, z, label, color=self.text_color, fontsize=10)
        
        # Set labels and title
        ax.set_xlabel('X', color=self.text_color)
        ax.set_ylabel('Y', color=self.text_color)
        ax.set_zlabel('Z', color=self.text_color)
        ax.set_title(f'3D Factorization Visualization of {number}', color=self.text_color, fontsize=16)
        
        # Add factorization equation
        factors_str = ' × '.join(map(str, factors))
        equation = f"{number} = {factors_str}"
        fig.text(0.5, 0.02, equation, ha='center', color=self.text_color, fontsize=14)
        
        # Make the visualization more dynamic by adding a slight rotation
        for angle in range(0, 360, 5):
            ax.view_init(elev=20, azim=angle)
            if angle == 30:  # Save at a specific angle
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        plt.close(fig)
        
        return filepath
    
    def _generate_3d_coordinates(self, number, factors):
        """Generate 3D coordinates for the factorization visualization."""
        factor_count = len(factors)
        
        # Coordinates for the original number (at center)
        x_coords = [0]
        y_coords = [0]
        z_coords = [0]
        colors = [self.node_colors[0]]
        sizes = [500]  # Larger size for the original number
        
        # Determine coordinates for factors
        if factor_count > 0:
            # Use prime factor coordinates
            if HAS_SEABORN:
                # Use t-SNE or similar for more interesting positioning
                prime_positions = {}
                for i, factor in enumerate(set(factors)):
                    # Map each prime to a position based on its properties
                    angle = (factor % 10) / 10 * 2 * np.pi
                    radius = 2 + (factor % 7) / 7
                    height = (factor % 5) / 5 * 2
                    
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    z = height
                    
                    prime_positions[factor] = (x, y, z)
                
                # Add coordinates for each factor
                for factor in factors:
                    pos = prime_positions[factor]
                    x_coords.append(pos[0])
                    y_coords.append(pos[1])
                    z_coords.append(pos[2])
                    
                    # Color based on primality and factor size
                    if self._is_prime(factor):
                        color_idx = min(factor % len(self.node_colors), len(self.node_colors) - 1)
                        colors.append(self.highlight_color if factor > 100 else self.node_colors[color_idx])
                    else:
                        colors.append(self.edge_color)
                    
                    # Size based on factor's magnitude
                    size = min(300, max(50, 200 - factor))
                    sizes.append(size)
            else:
                # Simple positioning on a sphere
                for i, factor in enumerate(factors):
                    # Position factors on a sphere around the original number
                    phi = np.pi * (1 + 5**0.5) * i  # Golden angle
                    theta = np.arccos(1 - 2 * ((i+1) / (factor_count+1)))
                    
                    r = 5  # Radius of the sphere
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)
                    
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)
                    
                    # Color based on primality
                    if self._is_prime(factor):
                        color_idx = min(i % len(self.node_colors), len(self.node_colors) - 1)
                        colors.append(self.highlight_color if factor > 100 else self.node_colors[color_idx])
                    else:
                        colors.append(self.edge_color)
                    
                    # Size based on factor's magnitude
                    size = min(300, max(50, 200 - factor))
                    sizes.append(size)
        
        return x_coords, y_coords, z_coords, colors, sizes
    
    def visualize_factorization_heatmap(self, number: int, factors: List[int] = None,
                                      filename: str = None) -> str:
        """
        Create a heatmap visualization of the factorization structure.
        
        Args:
            number: The number being factorized
            factors: Optional list of prime factors (if not provided, will be calculated)
            filename: Optional filename to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        # Calculate factors if not provided
        if factors is None:
            factors = self.factorize(number)
        
        # Use default filename if not provided
        if filename is None:
            filename = f"factorization_heatmap_{number}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create a factorization matrix
        matrix, labels = self._generate_factor_matrix(number, factors)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)
        
        # Create heatmap
        if HAS_SEABORN:
            palette = "viridis_r" if self.dark_mode else "viridis"
            sns.heatmap(matrix, annot=True, fmt=".2g", linewidths=0.5, cmap=palette,
                      xticklabels=labels, yticklabels=labels, ax=ax)
        else:
            im = ax.imshow(matrix, cmap='viridis')
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            
            # Add text annotations
            for i in range(len(labels)):
                for j in range(len(labels)):
                    text = ax.text(j, i, f"{matrix[i, j]:.2g}",
                                 ha="center", va="center", color="white")
        
        # Set title and labels
        ax.set_title(f"Factorization Structure of {number}", color=self.text_color, fontsize=16)
        
        # Add factorization equation
        factors_str = ' × '.join(map(str, factors))
        equation = f"{number} = {factors_str}"
        fig.text(0.5, 0.02, equation, ha='center', color=self.text_color, fontsize=14)
        
        # Adjust ticks
        ax.tick_params(colors=self.text_color)
        plt.xticks(rotation=45, ha='right')
        
        # Save and return
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
    
    def _generate_factor_matrix(self, number, factors):
        """Generate a matrix representation of the factorization structure."""
        # Create labels for original number and all unique factors
        unique_factors = sorted(set(factors))
        labels = [number] + unique_factors
        n = len(labels)
        
        # Initialize matrix
        matrix = np.zeros((n, n))
        
        # Fill the matrix based on factorization relationships
        for i, val_i in enumerate(labels):
            for j, val_j in enumerate(labels):
                if i == j:
                    # Self-relation (diagonal): use bit length
                    matrix[i, j] = val_i.bit_length() if isinstance(val_i, int) else 0
                elif i == 0 and val_j in factors:
                    # Relationship between original number and its factors
                    matrix[i, j] = factors.count(val_j)
                elif j == 0 and val_i in factors:
                    # Symmetric relationship
                    matrix[i, j] = factors.count(val_i)
                elif val_i != number and val_j != number:
                    # Relationship between factors: co-occurrence pattern
                    if val_i % val_j == 0 or val_j % val_i == 0:
                        matrix[i, j] = 0.5
                    if self._is_prime(val_i) and self._is_prime(val_j):
                        # Prime-to-prime relationship: use arithmetic properties
                        matrix[i, j] = 1.0 / (abs(val_i - val_j) + 1)
        
        return matrix, labels
    
    def factorize(self, n):
        """Simple factorization function for demonstration."""
        if n <= 1:
            return [n]
        
        factors = []
        d = 2
        
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1 if d == 2 else 2
        
        if n > 1:
            factors.append(n)
        
        return factors


def generate_sample_visualizations():
    """Generate sample visualizations for demonstration."""
    # Create visualizers
    tree_vis = DecompositionTreeVisualizer(dark_mode=True)
    algo_vis = AlgorithmVisualizer(dark_mode=True)
    complex_vis = FactorizationTreeVisualizer(dark_mode=True)
    
    # Test numbers
    numbers = [
        30030,      # 2 × 3 × 5 × 7 × 11 × 13 (many small factors)
        104729873,  # Large semiprime
        2**16,      # Power of 2
        2**8 - 1    # Mersenne number
    ]
    
    print("Generating visualizations...")
    
    # Generate tree visualizations
    for number in numbers:
        print(f"Creating decomposition tree for {number}")
        tree_vis.visualize_decomposition_tree(number, show_steps=True)
    
    # Generate algorithm visualizations
    print("\nCreating algorithm visualizations...")
    algorithms = ["trial_division", "pollard_rho", "quadratic_sieve"]
    
    for algorithm in algorithms:
        print(f"Visualizing {algorithm} algorithm")
        algo_vis.visualize_algorithm(algorithm, numbers[0])
    
    # Generate complex visualizations
    print("\nCreating advanced visualizations...")
    for number in numbers[:2]:  # Just use the first two numbers
        print(f"Creating 3D visualization for {number}")
        complex_vis.visualize_3d_factorization(number)
        
        print(f"Creating heatmap for {number}")
        complex_vis.visualize_factorization_heatmap(number)
    
    print("\nAll visualizations generated in the 'visualizations' directory")


if __name__ == "__main__":
    generate_sample_visualizations()