import tensorflow
import pandas as pd
import io
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np

def get_dataframe():
    feature_engineered_aflow_df = pd.read_csv('/home/dts/Downloads/result_best_egap_model_tim_plus_features.csv')
    # We need to drop many columns:
    tim_aflow_df = pd.read_csv('/home/dts/Downloads/result_best_egap_model_tim.csv')
    # Merge the dataframes
    merged_df = pd.concat([tim_aflow_df.reset_index(drop=True),feature_engineered_aflow_df.reset_index(drop=True)], axis=1)
    # Now split the data into test and train:

    merged_df_removed_na = merged_df.dropna(
        axis = 1)

    merged_df_train = merged_df_removed_na[
        merged_df['split'] == 'train']
    merged_df_val = merged_df_removed_na[
        merged_df['split'] == 'validation']
    merged_df_test = merged_df_removed_na[
        merged_df['split'] == 'test']

    columns = [
        'spacegroup_relax',
        'energy_cutoff', 'density',
        'cutoff_val', 'n_edge',
        'EA_half_bar', 'EA_half_hat', 'IP_half_bar', 'IP_half_hat',
        'EA_delta_bar', 'EA_delta_hat', 'IP_delta_bar', 'IP_delta_hat',
        'HOMO_bar', 'HOMO_hat', 'LUMO_bar', 'LUMO_hat', 'rs_bar', 'rs_hat',
        's index_bar', 's index_hat', 'rp_bar', 'rp_hat', 'p index_bar',
        'p index_hat', 'rd_bar', 'rd_hat', 'd index_bar', 'd index_hat',
        'rf_bar', 'rf_hat', 'f index_bar', 'f index_hat', 'atomic_number_bar',
        'atomic_number_hat', 'atomic_weight_bar', 'atomic_weight_hat',
        'mendeleev_number_bar', 'mendeleev_number_hat', 'melting_point_bar',
        'melting_point_hat', 'covalent_radius_cordero_bar',
        'covalent_radius_cordero_hat',
        # 'en_allen_bar', 'en_allen_hat',
        'en_ghosh_bar', 'en_ghosh_hat',
        # 'en_pauling_bar', 'en_pauling_hat',
        'period_bar', 'period_hat',
        # 'atomic_volume_bar',
        # 'atomic_volume_hat',
        'n_valence_bar', 'n_valence_hat', 'HOMO_LUMO_diff_bar',
        'HOMO_LUMO_diff_hat',
        'Egap']

    merged_df_removed_bad_features_train = merged_df_train[
        columns]
    merged_df_removed_bad_features_test = merged_df_test[
        columns]
    merged_df_removed_bad_features_val = merged_df_val[
        columns]

    # Print the number of each
    training_length = len(merged_df_removed_bad_features_train["Egap"])
    test_length = len(merged_df_removed_bad_features_test["Egap"])
    print(f'# of training data points: {training_length}')
    print(f'# of test data points: {test_length}')
    print(
        f'training fraction is: {training_length/(training_length + test_length)}')
    return merged_df_removed_bad_features_train, merged_df_removed_bad_features_test, merged_df_removed_bad_features_val


def get_tf_data(
        merged_df_removed_bad_features_train,
        merged_df_removed_bad_features_val,
        merged_df_removed_bad_features_test):
    X_train_df = merged_df_removed_bad_features_train.drop(columns=['Egap'])
    y_train_df = merged_df_removed_bad_features_train[['Egap']]

    X_train = tf.convert_to_tensor(X_train_df)
    y_train = tf.convert_to_tensor(y_train_df)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train_df)) # Convert to numpy array

    X_val_df = merged_df_removed_bad_features_val.drop(
        columns=['Egap'])
    y_val_df = merged_df_removed_bad_features_val[
        ['Egap']]

    X_val = tf.convert_to_tensor(X_val_df)
    y_val = tf.convert_to_tensor(y_val_df)


    X_test_df = merged_df_removed_bad_features_test.drop(
        columns=['Egap'])
    y_test_df = merged_df_removed_bad_features_test[
        ['Egap']]

    X_test = tf.convert_to_tensor(X_test_df)
    y_test = tf.convert_to_tensor(y_test_df)
    print(tf.shape(X_train))
    print(tf.shape(X_test))
    normalizer(X_train_df.iloc[:3])
    return normalizer, X_train, X_test, X_val, y_train, y_test, y_val


def create_elemnet_model_with_residuals(normalizer, input_features_shape):
    # Input layer (after normalization)
    # --- Block 1 (1024 neurons) ---
    # Store the input to this block for the residual connection
    input_tensor = layers.Input(shape=input_features_shape) # Get shape from normalizer output
    x = normalizer(input_tensor) # Apply normalization first
    residual_input_1 = input_tensor
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    # Residual connection: add residual_input_1 to x
    # Ensure dimensions match. If residual_input_1 had a different shape,
    # we'd need a Dense layer on it to project to 1024.
    # Here, 'x' already has 1024 neurons, so we need to project 'residual_input_1' to 1024 if its original shape was different.
    # Assuming normalizer output matches 1024 for the first block's input:
    # If the normalizer output is NOT 1024, use this:
    if residual_input_1.shape[-1] != 1024:
        residual_input_1 = layers.Dense(1024)(residual_input_1) # Project to match dimensions
    x = layers.Add()([x, residual_input_1])
    # Typically, activation comes AFTER the addition in ResNets
    x = layers.Activation("relu")(x) # Apply activation after adding residual

    # --- Block 2 (512 neurons) ---
    residual_input_2 = x # Store input for this block
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    # Residual connection
    if residual_input_2.shape[-1] != 512:
        residual_input_2 = layers.Dense(512)(residual_input_2) # Project to match dimensions
    x = layers.Add()([x, residual_input_2])
    x = layers.Activation("relu")(x)

    # --- Block 3 (256 neurons) ---
    residual_input_3 = x # Store input for this block
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    # Residual connection
    if residual_input_3.shape[-1] != 256:
        residual_input_3 = layers.Dense(256)(residual_input_3) # Project to match dimensions
    x = layers.Add()([x, residual_input_3])
    x = layers.Activation("relu")(x)

    # --- Block 4 (128 neurons) ---
    residual_input_4 = x # Store input for this block
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    # Residual connection
    if residual_input_4.shape[-1] != 128:
        residual_input_4 = layers.Dense(128)(residual_input_4) # Project to match dimensions
    x = layers.Add()([x, residual_input_4])
    x = layers.Activation("relu")(x)

    # --- Final Layers (without explicit residual blocks here, can be added if desired) ---
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    output_tensor = layers.Dense(1)(x) # Output layer (linear activation by default)

    # Create the functional model
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


def save_model(model, model_name: str):
    local_save_dir = '/home/dts/Documents/theory/thesis/models'
    os.makedirs(local_save_dir, exist_ok=True) # Create the directory if it doesn't exist
    print(f"\nModels will be saved to: {os.path.abspath(local_save_dir)}")

    # --- 5. Save the Trained Model Locally ---

    # Option A: TensorFlow SavedModel format (recommended)
    model_path_tf = os.path.join(local_save_dir, model_name)

    print(f"\n--- Saving model to {model_path_tf} ---")
    model.save(model_path_tf)
    print("Model saved successfully in TensorFlow SavedModel format!")
    return model_path_tf



# --- Example Usage ---
if __name__ == "__main__":

    (merged_df_removed_bad_features_train,
     merged_df_removed_bad_features_test,
     merged_df_removed_bad_features_val) = get_dataframe()
    normalizer, X_train, X_test, X_val, y_train, y_test, y_val = get_tf_data(
        merged_df_removed_bad_features_train,
        merged_df_removed_bad_features_test,
        merged_df_removed_bad_features_val)
    input_features_shape = X_train.shape[1:] # e.g., (50,) if X_train_df is (100, 50)
    model_with_res = create_elemnet_model_with_residuals(normalizer, input_features_shape)
    model_with_res.summary()
    # You can now compile and train this model
    learning_rate = 1E-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model_with_res.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    EPOCHS = 1
    model_with_res.fit(X_train, y_train, epochs=EPOCHS, batch_size=32)
    model_path_tf = save_model(model_with_res, 'elemnet_resnet')
    # Load the model 
    loaded_model_tf = tf.keras.models.load_model(model_path_tf)
    print(f"Model loaded successfully from {model_path_tf}")
    loaded_model_tf.summary()

    # Make a prediction to test
    print("\n--- Making a prediction with the loaded SavedModel ---")
    # dummy_test_data = np.random.rand(5, input_features_shape) * 100 # 5 new samples
    predictions_tf = loaded_model_tf.predict(X_test)
    print(f"Predictions (SavedModel): {predictions_tf.flatten()}")


    

    # print("\n--- Original Sequential Model (for comparison) ---")
    # original_model = tf.keras.Sequential([
    #     normalizer,
    #     layers.Dense(1024, activation="relu"),
    #     layers.Dense(1024, activation="relu"),
    #     layers.Dense(1024, activation="relu"),
    #     layers.Dense(1024, activation="relu"),
    #     layers.Dropout(0.2),
    #     layers.Dense(512, activation="relu"),
    #     layers.Dense(512, activation="relu"),
    #     layers.Dense(512, activation="relu"),
    #     layers.Dropout(0.1),
    #     layers.Dense(256, activation="relu"),
    #     layers.Dense(256, activation="relu"),
    #     layers.Dense(256, activation="relu"),
    #     layers.Dropout(0.3),
    #     layers.Dense(128, activation="relu"),
    #     layers.Dense(128, activation="relu"),
    #     layers.Dense(128, activation="relu"),
    #     layers.Dropout(0.2),
    #     layers.Dense(64, activation="relu"),
    #     layers.Dense(64, activation="relu"),
    #     layers.Dense(32, activation="relu"),
    #     layers.Dense(1)
    # ])
    # original_model.build(input_shape=(None, input_features_shape)) # Need to build for summary
    # original_model.summary()

