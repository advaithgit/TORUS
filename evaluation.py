import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats
from typing import Dict, List
from sklearn.utils import resample

class StatisticalValidator:
    """Handles statistical significance testing"""
    
    def __init__(self, results: Dict):
        self.results = results

    def calculate_pairwise_tests(self) -> Dict:
        """Calculate paired t-tests against all baselines"""
        report = {}
        for load_level in self.results['our_model']:
            our_data = self._extract_metrics('our_model', load_level)
            baseline_comparisons = {}
            for baseline in ['XY', 'Adaptive', 'Q-Routing', 'Attention']:
                base_data = self._extract_metrics(baseline, load_level)
                if len(base_data) > 1 and len(our_data) > 1:  # Require ≥2 samples
                    t_stat, p_val = stats.ttest_rel(our_data, base_data)
                    baseline_comparisons[baseline] = {
                        't-statistic': t_stat,
                        'p-value': p_val,
                        'significant': p_val < 0.05
                    }
            report[load_level] = baseline_comparisons
        return report

    def bootstrap_confidence_intervals(self, metric: str, n_boot: int = 1000) -> Dict:
        """Calculate bootstrap CIs for all models"""
        ci_report = {}
        for model in self.results:
            model_data = []
            for load in self.results[model]:
                model_data.extend(self._extract_metrics(model, load, metric))
            boot_means = []
            for _ in range(n_boot):
                sample = resample(model_data, replace=True)
                boot_means.append(np.mean(sample))
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)
            ci_report[model] = (ci_low, ci_high)
        return ci_report

    def _extract_metrics(self, model: str, load_level: float, metric: str = 'success_rate') -> List[float]:
        """Helper to extract specific metric values"""
        return [p[metric] for p in self.results[model][load_level]]

class PerformanceVisualizer:
    """Creates interactive and static visualizations"""
    
    def __init__(self, results: Dict):
        self.results = results
        self.colors = {
            'our_model': '#1f77b4',
            'XY': '#ff7f0e',
            'Adaptive': '#2ca02c',
            'Q-Routing': '#d62728',
            'Attention': '#9467bd'
        }

    def plot_metric_trends(self, metric: str, interactive: bool = True) -> None:
        """Plot metric across load levels for all models"""
        load_levels = sorted(self.results['our_model'].keys(), reverse=True)
        
        if interactive:
            fig = go.Figure()
            for model in self.results:
                metric_values = [np.mean(self.results[model][ll][metric]) 
                                  for ll in load_levels]
                fig.add_trace(go.Scatter(
                    x=load_levels,
                    y=metric_values,
                    name=model,
                    line=dict(color=self.colors[model]),
                    mode='lines+markers'
                ))
            fig.update_layout(
                title=f'{metric.replace("_", " ").title()} Comparison',
                xaxis_title='Network Load Level',
                yaxis_title=metric.replace('_', ' ').title(),
                showlegend=True
            )
            fig.show()
        else:
            plt.figure(figsize=(10, 6))
            for model in self.results:
                metric_values = [np.mean(self.results[model][ll][metric]) 
                                  for ll in load_levels]
                plt.plot(load_levels, metric_values,
                         color=self.colors[model],
                         marker='o',
                         label=model)
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.xlabel('Network Load Level')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.legend()
            plt.grid(True)
            plt.show()

    def generate_comparison_report(self, stats_report: Dict) -> str:
        """Generate markdown format comparison report"""
        report = ["# Torus Network Routing Performance Report\n"]
        # Success Rate Analysis
        report.append("## Success Rate Comparison\n")
        report.append(self._metric_table('success_rate', stats_report))
        # Contention Analysis
        report.append("\n## Contention Events Comparison\n")
        report.append(self._metric_table('contention_events', stats_report))
        return '\n'.join(report)

    def _metric_table(self, metric: str, stats_report: Dict) -> str:
        table = [
            "| Load Level | Our Model | XY Routing | Adaptive | Q-Routing | Attention |",
            "|------------|-----------|------------|----------|-----------|-----------|"
        ]
        for load in sorted(stats_report.keys(), reverse=True):
            row = [f"| {load*100:.0f}% "]
            for model in ['our_model', 'XY', 'Adaptive', 'Q-Routing', 'Attention']:
                if model == 'our_model':
                    mean_val = np.mean(self.results[model][load][metric])
                    row.append(f"{mean_val:.2f} ")
                else:
                    mean_val = np.mean(self.results[model][load][metric])
                    p_val = stats_report[load][model]['p-value']
                    sig = '*' if stats_report[load][model]['significant'] else ''
                    row.append(f"{mean_val:.2f}{sig} ")
            table.append('|'.join(row) + '|')
        return '\n'.join(table)

class NetworkValidator:
    """Main validation workflow controller"""
    
    def __init__(self, test_dataset: pd.DataFrame):
        self.dataset = test_dataset
        self.baselines = ['XY', 'Adaptive', 'Q-Routing', 'Attention']

    def calculate_metrics(self) -> Dict:
        """Calculate all metrics for all models"""
        results = {model: {} for model in ['our_model'] + self.baselines}
        for model in results:
            model_data = self.dataset[self.dataset['routing_algorithm'] == model]
            for load_level in model_data['load_level'].unique():
                subset = model_data[model_data['load_level'] == load_level]
                results[model][load_level] = {
                    'success_rate': subset['success'].mean(),
                    'avg_contention': subset['contention'].mean(),
                    'avg_time': subset['transmission_time'].mean(),
                    'reroute_percent': subset['reroute_percent'].mean()
                }
        return results

    def validate_against_baselines(self) -> Dict:
        results = self.calculate_metrics()
        validator = StatisticalValidator(results)
        return {
            'raw_metrics': results,
            'statistical_significance': validator.calculate_pairwise_tests(),
            'confidence_intervals': validator.bootstrap_confidence_intervals('success_rate')
        }

# After running simulations
validator = NetworkValidator(pd.DataFrame())  # Placeholder for actual dataset
results = validator.validate_against_baselines()

# Generate visualizations
visualizer = PerformanceVisualizer(results['raw_metrics'])
visualizer.plot_metric_trends('success_rate', interactive=True)

# Generate report
report = visualizer.generate_comparison_report(results['statistical_significance'])
with open('performance_report.md', 'w') as f:
    f.write(report)