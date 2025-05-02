import argparse
import pandas as pd
from .simulator import EnhancedNetworkSimulator
from .features import FeatureEngineer, DatasetSplitter
from .trainer import TorusTrainer
from .model import EnhancedTorusNet
from .config import TrafficPattern, PriorityScheme, TRAINING_CONFIG

def main():
    parser = argparse.ArgumentParser(description='Torus Network Simulation and Training')
    parser.add_argument('--simulate', action='store_true', help='Run network simulation')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--load', type=float, default=1.0, help='Network load level (0.1-1.0)')
    args = parser.parse_args()

    if args.simulate:
        # Run simulation and save dataset
        sim = EnhancedNetworkSimulator()
        sim.run_simulation(
            load_level=args.load,
            duration=1000,
            pattern=TrafficPattern.UNIFORM,
            priority_scheme=PriorityScheme.HYBRID
        )
        engineer = FeatureEngineer(sim)
        dataset = engineer.create_ml_dataset()
        dataset.to_parquet(f"dataset_{args.load}.parquet")
        print(f"Saved dataset for load {args.load}")

    if args.train:
        # Load dataset and train model
        try:
            dataset = pd.read_parquet(f"dataset_{args.load}.parquet")
        except FileNotFoundError:
            raise ValueError(f"No dataset found for load {args.load}. Run simulation first.")
        
        splitter = DatasetSplitter(dataset)
        splits = splitter.load_stratified_split()
        
        model = EnhancedTorusNet()
        trainer = TorusTrainer(model, splits['train'], splits['val'])
        trainer.train(epochs=TRAINING_CONFIG['max_epochs'])
        print("Training completed")

if __name__ == "__main__":
    main()