import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

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
    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_momentum = False

    improved_weight_model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        improved_weight_model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    improved_weight_train_history, improved_weight_val_history = trainer.train(num_epochs)


# ______________________________________________________________________________________
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False

    improved_sigmoid_model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        improved_sigmoid_model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    improved_sigmoid_train_history, improved_sigmoid_val_history = trainer.train(num_epochs)


# ______________________________________________________________________________________
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    learning_rate = .02

    momentum_model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        momentum_model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    momentum_train_history, momentum_val_history = trainer.train(num_epochs)

# ______________________________________________________________________________________


    plt.figure(figsize=(20, 12))
    plt.subplot(2, 2, 1)
    plt.ylim([0., .5])
    utils.plot_loss(train_history["loss"], "Training Loss", npoints_to_average=10)
    utils.plot_loss(improved_weight_train_history["loss"], "Training Loss improved weight", npoints_to_average=10)
    utils.plot_loss(improved_sigmoid_train_history["loss"], "Training Loss improved sigmoid", npoints_to_average=10)
    utils.plot_loss(momentum_train_history["loss"], "Training Loss with momentum", npoints_to_average=10)
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")


    plt.subplot(2, 2, 2)
    plt.ylim([0., .5])
    utils.plot_loss(val_history["loss"], "Validation Loss")
    utils.plot_loss(improved_weight_val_history["loss"], "Validation Loss improved weight")
    utils.plot_loss(improved_sigmoid_val_history["loss"], "Validation Loss improved sigmoid")
    utils.plot_loss(momentum_val_history["loss"], "Validation Loss with momentum")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")

    # Plot accuracy
    plt.subplot(2, 2, 3)
    plt.ylim([0.90, 1.00])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(improved_weight_train_history["accuracy"], "Training Accuracy improved weight")
    utils.plot_loss(improved_sigmoid_train_history["accuracy"], "Training Accuracy improved sigmoid")
    utils.plot_loss(momentum_train_history["accuracy"], "Training Accuracy with momentum")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()


    plt.subplot(2, 2, 4)
    plt.ylim([0.90, 1.00])
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    utils.plot_loss(improved_weight_val_history["accuracy"], "Validation Accuracy improved weight")
    utils.plot_loss(improved_sigmoid_val_history["accuracy"], "Validation Accuracy improved sigmoid")
    utils.plot_loss(momentum_val_history["accuracy"], "Validation Accuracy with momentum")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("task3.png")




