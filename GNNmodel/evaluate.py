import torch
import logging
import numpy as np
from module import TGAN  # Make sure to import your model's class correctly

def main():
    # Constants (replace these with the correct paths or values)
    MODEL_SAVE_PATH = "/saved_models/-attn-prod-wet.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # or set a specific GPU

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Load the model
    
    model = load_model(MODEL_SAVE_PATH, DEVICE)
    logger.info("Model loaded successfully.")

    # Evaluate the model
    evaluate_model(model, DEVICE, logger)


def load_model(model_path, device):
    """
    Load the pre-trained model from the specified path.
    """
    # Make sure the TGAN initialization matches the one used during training
    model = TGAN()  # Add the necessary arguments for your model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# Import any other metrics or utilities you need

def evaluate_model(model, device, logger):
    """
    Evaluate the model on the validation/test dataset.
    """
    # Assuming you have a function 'create_your_data_loader' to prepare your dataset
    data_loader = create_your_data_loader()  # Replace with your function to get a DataLoader

    # These will store the predictions and true labels from all batches,
    # so you can calculate metrics at the end
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():  # No need to track gradients when evaluating
        model.eval()  # Set the model to evaluation mode

        # Go through each batch in the DataLoader
        for inputs, true_labels in data_loader:
            inputs = inputs.to(device)  # Move the inputs to the correct device
            true_labels = true_labels.to(device)  # Same for the labels

            # Get predictions
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)  # Assuming classification task

            # Store predictions and true labels for later
            all_predictions.append(predictions.cpu().numpy())
            all_true_labels.append(true_labels.cpu().numpy())

    # Convert lists to single numpy arrays for easier calculation
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    # Calculate metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average='macro')  # Adjust as needed
    recall = recall_score(all_true_labels, all_predictions, average='macro')  # Adjust as needed
    f1 = f1_score(all_true_labels, all_predictions, average='macro')  # Adjust as needed
    auc = roc_auc_score(all_true_labels, all_predictions, average='macro', multi_class='ovr')  # Adjust as needed

    # Log or print your evaluation results
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"AUC: {auc}")

    # You can return the metrics if you plan to use them later
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

# Replace 'create_your_data_loader' with the actual function or process you use to load your data for evaluation.
def create_your_data_loader():
    # Implement or call your actual data loading mechanism here
    pass

if __name__ == "__main__":
    main()
