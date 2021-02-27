import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    #neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

# ______________________________________________________________________________________
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    learning_rate = .02

    neurons_per_layer = [64, 10]

    original_model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        original_model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    original_train_history, original_val_history = trainer.train(num_epochs)
    
    parameters = sum([w.size for w in original_model.ws])
    print("Network {} has {} parameters".format(neurons_per_layer, parameters))

# ______________________________________________________________________________________
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    learning_rate = .02

    neurons_per_layer = [60, 60, 10]

    three_layer_model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        three_layer_model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    three_layer_train_history, three_layer_val_history = trainer.train(num_epochs)

    parameters = sum([w.size for w in three_layer_model.ws])
    print("Network {} has {} parameters".format(neurons_per_layer, parameters))

# ______________________________________________________________________________________
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    learning_rate = .02

    #neurons_per_layer = [64]*10+[10]
    neurons_per_layer = [300, 300, 10]


    expanded_model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        expanded_model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    expanded_train_history, expanded_val_history = trainer.train(num_epochs)

    parameters = sum([w.size for w in expanded_model.ws])
    print("Network {} has {} parameters".format(neurons_per_layer, parameters))

# ______________________________________________________________________________________

    # Plot loss for first model
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 2, 1)
    plt.ylim([0., .5])
    utils.plot_loss(original_train_history["loss"], "Training Loss (layers 64-10)", npoints_to_average=10)
    utils.plot_loss(three_layer_train_history["loss"], "Training Loss (layers 60-60-10)", npoints_to_average=10)
    utils.plot_loss(expanded_train_history["loss"], "Training Loss (layers 64x10-10)", npoints_to_average=10)
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Loss - Average")

    plt.subplot(2, 2, 2)
    plt.ylim([0., .5])
    utils.plot_loss(original_val_history["loss"], "Validation Loss (layers 64-10)")
    utils.plot_loss(three_layer_val_history["loss"], "Validation Loss (layers 60-60-10)")
    utils.plot_loss(expanded_val_history["loss"], "Validation Loss (layers 64x10-10)")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Loss - Average")

    # Plot accuracy
    plt.subplot(2, 2, 3)
    plt.ylim([0.90, 1.00])
    utils.plot_loss(original_train_history["accuracy"], "Training Accuracy (layers 64-10)")
    utils.plot_loss(three_layer_train_history["accuracy"], "Training Accuracy (layers 60-60-10)")
    utils.plot_loss(expanded_train_history["accuracy"], "Training Accuracy (layers 64x10-10)")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Accuracy")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.ylim([0.90, 1.00])
    utils.plot_loss(original_val_history["accuracy"], "Validation Accuracy (layers 64-10)")
    utils.plot_loss(three_layer_val_history["accuracy"], "Validation Accuracy (layers 60-60-10)")
    utils.plot_loss(expanded_val_history["accuracy"], "Validation Accuracy (layers 64x10-10)")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()

    plt.savefig("task4e.png")



