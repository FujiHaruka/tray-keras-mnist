from dataset import load_dataset
from model import create_cnn_model

def learn(batch_size=32, epochs=10):
    X_train, X_test, Y_train, Y_test, input_shape = load_dataset()

    model = create_cnn_model(input_shape)

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, Y_test))

if __name__ == '__main__':
    learn()
