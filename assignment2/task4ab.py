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

    neurons_per_layer = [32, 10]

    fewer_layers_model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        fewer_layers_model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    fewer_layers_train_history, fewer_layers_val_history = trainer.train(num_epochs)

# ______________________________________________________________________________________
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    learning_rate = .02

    neurons_per_layer = [128, 10]

    more_layers_model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        more_layers_model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    more_layers_train_history, more_layers_val_history = trainer.train(num_epochs)
# ______________________________________________________________________________________
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    learning_rate = .02

    neurons_per_layer = [64, 10]

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
# ______________________________________________________________________________________

    # Plot loss for first model (task 2c)
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 2, 1)
    plt.ylim([0., .5])
    utils.plot_loss(fewer_layers_train_history["loss"], "Training Loss (32 hidden units)", npoints_to_average=10)
    utils.plot_loss(train_history["loss"], "Training Loss (64 hidden units)", npoints_to_average=10)
    utils.plot_loss(more_layers_train_history["loss"], "Training Loss (128 hidden units)", npoints_to_average=10)
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Loss - Average")

    plt.subplot(2, 2, 2)
    plt.ylim([0., .5])
    utils.plot_loss(fewer_layers_val_history["loss"], "Validation Loss (32 hidden units)")
    utils.plot_loss(val_history["loss"], "Validation Loss (64 hidden units)")
    utils.plot_loss(more_layers_val_history["loss"], "Validation Loss (128 hidden units)")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Loss - Average")

    # Plot accuracy
    plt.subplot(2, 2, 3)
    plt.ylim([0.90, 1.00])
    utils.plot_loss(fewer_layers_train_history["accuracy"], "Training Accuracy (32 hidden units)")
    utils.plot_loss(train_history["accuracy"], "Training Accuracy (64 hidden units)")
    utils.plot_loss(more_layers_train_history["accuracy"], "Training Accuracy (128 hidden units)")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Accuracy")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.ylim([0.90, 1.00])
    utils.plot_loss(fewer_layers_val_history["accuracy"], "Validation Accuracy (32 hidden units)")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy (64 hidden units)")
    utils.plot_loss(more_layers_val_history["accuracy"], "Validation Accuracy (128 hidden units)")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()

    plt.savefig("task4ab.png")



