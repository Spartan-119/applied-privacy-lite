from collections import OrderedDict
from omegaconf import DictConfig
import torch
from model import ConvolutionNeuralNet, test_model

def prepare_configuration(config: DictConfig):
    """
    Return function that prepares configuration to send to clients.

    Args:
        config: Configuration settings.

    Returns:
        Function to configure fit settings for clients during federated learning.
    """
    def fit_config_fn(server_round: int):
        """
        Configure fit settings for clients during federated learning.

        Args:
            server_round: Current round number in the federated learning process.

        Returns:
            Dictionary with fit configuration settings for clients.
        """
        # This function can be adapted over time based on server_round.
        return {
            "lr": config.learning_rate,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn

def get_global_evaluation(num_classes: int, testloader):
    """
    Define function for global evaluation on the server.

    Args:
        num_classes: Number of classes in the dataset.
        testloader: DataLoader for the test dataset.

    Returns:
        Function to evaluate the global model on the server.
    """
    def evaluate_fn(server_round: int, parameters, config):
        """
        Evaluate the global model on the test dataset.

        Args:
            server_round: Current round number in the federated learning process.
            parameters: Parameters of the global model.
            config: Configuration settings.

        Returns:
            Tuple containing loss and a dictionary with evaluation metrics.
        """
        # Initialize model and device
        model = ConvolutionNeuralNet(num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load parameters into the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({key: torch.Tensor(value) for key, value in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Evaluate the global model on the test set
        loss, accuracy = test_model(model, testloader, device)

        # Report the loss and any other metric (inside a dictionary)
        return loss, {"accuracy": accuracy}

    return evaluate_fn