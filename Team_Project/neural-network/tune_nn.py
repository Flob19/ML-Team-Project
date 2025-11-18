import keras_tuner as kt
from tensorflow.keras import layers, models, optimizers

def build_model(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train_processed.shape[1],)))

    # Tune number of layers
    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation='relu'
        ))
        model.add(layers.Dropout(hp.Float(f'dropout_{i}', 0.0, 0.4, step=0.1)))

    model.add(layers.Dense(1, activation='linear'))

    # Tune learning rate
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=['mae'])
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_mae',
    max_trials=15,
    executions_per_trial=1,
    directory='results/tuning',
    project_name='coffee_price_nn'
)

tuner.search(X_train_processed, y_train,
             validation_data=(X_test_processed, y_test),
             epochs=50,
             batch_size=32,
             callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

best_model = tuner.get_best_models(num_models=1)[0]
loss, mae = best_model.evaluate(X_test_processed, y_test)
print(f"Best tuned model Test MAE: {mae:.3f}")
