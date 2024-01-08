import pickle
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from dataset import prepare_federated_dataset
from client import generate_client_function
from server import prepare_configuration, get_global_evaluation

@hydra.main(config_path = "conf", config_name = "base", version_base = None)
def main(config: DictConfig):
    """Main function to run federated learning simulation."""

    # Parse config and get experiment output dir
    print(OmegaConf.to_yaml(config))
    save_path = HydraConfig.get().runtime.output_dir

    # Prepare teh federated dataset
    trainloaders, validationloaders, testloader = prepare_federated_dataset(
        config.number_of_clients, config.client_batch_size
    )

    # Define client function
    client_fn = generate_client_function(trainloaders, validationloaders, config.number_of_classes)

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit = 0.0,
        min_fit_clients = config.number_of_clients_per_round_fit,
        fraction_evaluate = 0.0,
        min_evaluate_clients = config.number_of_clients_per_round_eval,
        min_available_clients = config.number_of_clients,
        on_fit_config_fn = prepare_configuration(config.configuration_fit),
        evaluate_fn = get_global_evaluation(config.number_of_classes, testloader),
    )

    # Start Simulation
    history = fl.simulation.start_simulation(
        client_fn = client_fn,
        num_clients = config.number_of_clients,
        config=fl.server.ServerConfig(num_rounds = config.number_of_rounds),
        strategy = strategy,
        client_resources = {
            "num_cpus": 2,
            "num_gpus": 0.0,
        },
    )

    # Save results
    results_path = Path(save_path) / "results.pkl"
    results = {"history": history, "anythingelse": "here"}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()