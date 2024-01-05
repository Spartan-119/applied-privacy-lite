from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client for Federated Learning."""

    def __init__(self, trainloader, valloader, num_classes) -> None:
        """
        Initialize FlowerClient.

        Args:
            trainloader: DataLoader for training data.
            valloader: DataLoader for validation data.
            num_classes: Number of classes in the dataset.
        """
        super().__init__()

        # Dataloaders pointing to the data associated with this client
        self.trainloader = trainloader
        self.valloader = valloader

        # A model that is randomly initialized at first
        self.model = Net(num_classes)

        # Determine if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        """
        Receive parameters and apply them to the local model.

        Args:
            parameters: Model parameters received from the server.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """
        Extract model parameters and return them as a list of numpy arrays.

        Args:
            config: Configuration dictionary.

        Returns:
            List of numpy arrays representing model parameters.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """
        Train the model using local data and send it back to the server.

        Args:
            parameters: Model parameters received from the server.
            config: Configuration dictionary with hyperparameters.

        Returns:
            Tuple containing updated model parameters, number of examples, and metrics.
        """
        # Copy parameters sent by the server into the client's local model
        self.set_parameters(parameters)

        # Extract hyperparameters from the config sent by the server
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        # Standard optimizer setup
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Local training
        train(self.model, self.trainloader, optim, epochs, self.device)

        # Return updated model parameters, number of examples, and metrics
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """
        Evaluate the model on the validation data.

        Args:
            parameters: Model parameters received from the server.
            config: Configuration dictionary.

        Returns:
            Tuple containing loss, number of examples, and evaluation metrics.
        """
        # Set parameters received from the server
        self.set_parameters(parameters)

        # Perform evaluation
        loss, accuracy = test(self.model, self.valloader, self.device)

        # Return loss, number of examples, and evaluation metrics
        return float(loss), len(self.valloader), {"accuracy": accuracy}

def generate_client_fn(trainloaders, valloaders, num_classes):
    """
    Return a function to spawn a FlowerClient with specified train and validation loaders.

    Args:
        trainloaders: List of train loaders for each client.
        valloaders: List of validation loaders for each client.
        num_classes: Number of classes in the dataset.

    Returns:
        Function to spawn a FlowerClient with a given client ID.
    """
    def client_fn(cid: str):
        """
        Spawn a FlowerClient with the specified client ID.

        Args:
            cid: Client ID.

        Returns:
            FlowerClient using the specified train and validation loaders.
        """
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            num_classes=num_classes,
        )

    return client_fn
