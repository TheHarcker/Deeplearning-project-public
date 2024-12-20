"""
This script demonstrates a basic example of training a Physics-informed Neural Network (PaiNN)
to predict the QM9 property known as "internal energy at 0K". 
This property, like many others in the QM9 dataset, is computed as a sum of contributions from individual atoms.
"""

import torch  # Importing PyTorch for tensor operations and model training
import argparse  # Importing argparse for command line argument parsing
from tqdm import trange  # Importing tqdm for progress bar functionality
import torch.nn.functional as F  # Importing functional interface for neural network operations
from src.data import QM9DataModule  # Importing the data module for QM9 dataset handling
from pytorch_lightning import seed_everything  # Importing function to set random seed for reproducibility
from src.models import PaiNN, AtomwisePostProcessing  # Importing the model and post-processing class

def cli():
    """
    Command line interface to parse input arguments for the training script.
    
    Returns:
        args: Namespace containing the input arguments.
    """
    parser = argparse.ArgumentParser()  # Initializing argument parser
    parser.add_argument('--seed', default=0)  # Random seed for reproducibility

    # Data-related parameters
    parser.add_argument('--target', default=7, type=int)  # Set target property index; 7 corresponds to internal energy at 0K
    parser.add_argument('--data_dir', default='data/', type=str)  # Directory to the data files
    parser.add_argument('--batch_size_train', default=100, type=int)  # Batch size for training
    parser.add_argument('--batch_size_inference', default=1000, type=int)  # Batch size for inference
    parser.add_argument('--num_workers', default=0, type=int)  # Number of worker threads for data loading
    parser.add_argument('--splits', nargs=3, default=[110000, 10000, 10831], type=int)  # Number of samples for training, validation, and testing
    parser.add_argument('--subset_size', default=None, type=int)  # Optional size of a subset of the data

    # Model parameters
    parser.add_argument('--num_message_passing_layers', default=3, type=int)  # Number of message passing layers in the model
    parser.add_argument('--num_features', default=128, type=int)  # Number of features in the hidden layers
    parser.add_argument('--num_outputs', default=1, type=int)  # Number of output values (1 for predicting energy)
    parser.add_argument('--num_rbf_features', default=20, type=int)  # Number of radial basis function features
    parser.add_argument('--num_unique_atoms', default=100, type=int)  # Number of unique atom types considered in the dataset
    parser.add_argument('--cutoff_dist', default=5.0, type=float)  # Cutoff distance for interactions between atoms

    # Training parameters
    parser.add_argument('--lr', default=5e-4, type=float)  # Learning rate for the optimizer
    parser.add_argument('--weight_decay', default=0.01, type=float)  # Weight decay for regularization
    parser.add_argument('--num_epochs', default=1000, type=int)  # Total number of epochs for training
    parser.add_argument('--save_last_model', default=True, type=bool) # Save model with lowest validation loss

    # Validation parameters
    parser.add_argument('--validate', default=True, type=bool) # Do validation (required for the arguments below)
    parser.add_argument('--lr_decay', default=0.5, type=float) # Learning rate decay (set to 0 to turn off)
    parser.add_argument('--smoothing_factor', default=0.9, type=float) # Exponential smoothing factor of validation loss (requires lr_decay > 0)
    parser.add_argument('--early_stop_patience', default=30, type=int) # Early stopping patience
    parser.add_argument('--decay_patience', default=5, type=int) # Learning rate decay patience
    parser.add_argument('--save_best_model', default=True, type=bool) # Save model with lowest validation loss

    # Extra arguments to handle further experimentation
    parser.add_argument('--compile', default=False, type=bool)  # Whether to run torch.compile on the model
    parser.add_argument('--use_mps_if_available', default=False, type=bool)  # Whether to compute on metal performance shaders (only relevant when running on MacOS)
    parser.add_argument('--use_high_matmul_precision', default=False, type=bool)  # Use a higher precision float type which is recommended on some CPUs
    parser.add_argument('--job_id', default="", type=str)  # job_id for saving the model

    args = parser.parse_args()  # Parsing the arguments from the command line
    return args  # Returning the parsed arguments


# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience  # How many epochs to wait for improvement
        self.delta = delta        # Minimum change to qualify as an improvement
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss_epoch: float):
        if self.best_loss is None:
            self.best_loss = val_loss_epoch
        elif val_loss_epoch < self.best_loss - self.delta:
            self.best_loss = val_loss_epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False


def main():
    """
    Main function to execute the training process of the PaiNN model.
    """
    args = cli()  # Retrieve command line arguments
    seed_everything(args.seed)  # Set the random seed for reproducibility of results
    
    # Use GPU if available, otherwise fallback to CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif args.use_mps_if_available and torch.device('mps'):
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using device {device}')

    if args.use_high_matmul_precision:
        print("Using high precision floats for matrix multiplications")
        torch.set_float32_matmul_precision('high')

    if args.job_id != "":
        model_path = f'Results/{args.job_id}/'
    else: 
        model_path = 'Results/latest/'

    # Initialize the data module for QM9 dataset
    dm = QM9DataModule(
        target=args.target,
        data_dir=args.data_dir,
        batch_size_train=args.batch_size_train,
        batch_size_inference=args.batch_size_inference,
        num_workers=args.num_workers,
        splits=args.splits,
        seed=args.seed,
        subset_size=args.subset_size,
    )
    dm.prepare_data()  # Prepare the data (download, if necessary)
    dm.setup()  # Setup the training and validation/test splits
    # Get statistics for the target variable to normalize predictions
    y_mean, y_std, atom_refs = dm.get_target_stats(
        remove_atom_refs=True, divide_by_atoms=True
    )

    # Instantiate the PaiNN model with specified parameters
    painn = PaiNN(
        num_message_passing_layers=args.num_message_passing_layers,
        num_features=args.num_features,
        num_outputs=args.num_outputs, 
        num_rbf_features=args.num_rbf_features,
        num_unique_atoms=args.num_unique_atoms,
        cutoff_dist=args.cutoff_dist,
    )

    # Instantiate post-processing to convert atomic contributions to predicted property
    post_processing = AtomwisePostProcessing(
        args.num_outputs, y_mean, y_std, atom_refs
    )

    # Compile 
    if args.compile:
        painn = torch.compile(painn)
        post_processing = torch.compile(post_processing)
        print("Compiled PaiNN and AtomwisePostProcessing module")
    
    painn.to(device)  # Move the model to the appropriate device (GPU/CPU)
    post_processing.to(device)  # Move the post-processing module to the same device

    # Initialize the optimizer for training
    optimizer = torch.optim.AdamW(
        painn.parameters(),
        lr=args.lr,  # Learning rate
        weight_decay=args.weight_decay,  # Regularization parameter
    )

    # Training and validation loss for each epoch
    train_loss = []
    val_loss = []

    # Initialize the scheduler for training
    if args.validate:
        smoothed_val_loss = None

        if args.save_best_model:
            best_val_loss = float('inf')
        else:
            best_val_loss = 0
    
        if args.lr_decay > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, patience=args.decay_patience)  

        if args.early_stop_patience > 0:
            early_stop = EarlyStopping(patience=args.early_stop_patience)      


    pbar = trange(args.num_epochs)  # Initialize a progress bar for epochs
    for epoch in pbar:
        # -------------------- TRAIN --------------------
        #       ____                                     
        #  _||__|  |  ______   ______   ______   ______  
        # (        | |      | |      | |      | |      | 
        # /-()---() ~ ()--() ~ ()--() ~ ()--() ~ ()--()             

        painn.train()  # Set the model to training mode
        loss_epoch = 0.  # Initialize loss for the epoch

        # Iterate through batches of training data
        for batch in dm.train_dataloader():
            batch = batch.to(device)  # Move batch data to the appropriate device

            # Forward pass to compute atomic contributions
            atomic_contributions = painn(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch
            )
            # Apply post-processing to obtain predictions from atomic contributions
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            # Compute loss between predictions and true values
            loss_step = F.mse_loss(preds, batch.y, reduction='sum')

            # Normalize loss by the number of samples in the batch
            loss = loss_step / len(batch.y)
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update model parameters

            loss_epoch += loss_step.detach().item()  # Accumulate loss for the epoch

        # Average loss across all training data
        loss_epoch /= len(dm.data_train)
        pbar.set_postfix_str(f'Train loss: {loss_epoch:.3e}')  # Update progress bar with loss

        # Save training loss
        train_loss.append(loss_epoch)

        # ------------------- VALIDATION -------------------
        if args.validate:
            val_loss_epoch = 0
            painn.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient computation
                # Iterate through batches of test data
                for batch in dm.val_dataloader():
                    batch = batch.to(device)  # Move batch data to the appropriate device

                    # Forward pass to compute atomic contributions
                    atomic_contributions = painn(
                        atoms=batch.z,
                        atom_positions=batch.pos,
                        graph_indexes=batch.batch,
                    )
                    # Apply post-processing to obtain predictions from atomic contributions
                    preds = post_processing(
                        atoms=batch.z,
                        graph_indexes=batch.batch,
                        atomic_contributions=atomic_contributions,
                    )
                    # TODO: Should it be mse or mae loss?
                    # Accumulate Mean Square Error
                    val_loss_epoch += F.mse_loss(preds, batch.y, reduction='sum')

            # Average loss across all validation data
            val_loss_epoch /= len(dm.data_val)

            # Save validation loss
            val_loss.append(val_loss_epoch)

            # Exponential smoothing of validation loss
            if args.smoothing_factor > 0:
                if smoothed_val_loss is None:
                    smoothed_val_loss = val_loss_epoch
                else:
                    smoothed_val_loss = args.smoothing_factor*smoothed_val_loss + (1 - args.smoothing_factor)*val_loss_epoch
            else:
                smoothed_val_loss = val_loss_epoch

            # Save best model
            if smoothed_val_loss < best_val_loss:
                best_val_loss = smoothed_val_loss
                torch.save({
                    "painn": painn.state_dict(), 
                    "post_processing": post_processing.state_dict(),
                    "args": args,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, model_path + "best_model.pth")

            # Check for learning rate adjustment
            if args.lr_decay > 0:
                scheduler.step(smoothed_val_loss)

            # Early stopping
            if args.early_stop_patience > 0:
                if early_stop(smoothed_val_loss):
                    break

    # ------------------- TEST -------------------
    # Evaluation phase to compute Mean Absolute Error (MAE) on test data
    mae = 0
    painn.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        # Iterate through batches of test data
        for batch in dm.test_dataloader():
            batch = batch.to(device)  # Move batch data to the appropriate device

            # Forward pass to compute atomic contributions
            atomic_contributions = painn(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch,
            )
            # Apply post-processing to obtain predictions from atomic contributions
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            # Accumulate Mean Absolute Error
            mae += F.l1_loss(preds, batch.y, reduction='sum')

    # Average MAE across the entire test set
    mae /= len(dm.data_test)
    unit_conversion = dm.unit_conversion[args.target]  # Retrieve unit conversion function for the target
    print(f'Test MAE: {unit_conversion(mae):.3f}')  # Print the final MAE after unit conversion

    if args.save_last_model:
        torch.save({
            "painn": painn.state_dict(), 
            "post_processing": post_processing.state_dict(),
            "args": args,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        }, model_path + "last_model.pth")

if __name__ == '__main__':
    main()  # Execute the main function when the script is run
