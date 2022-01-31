import argparse
import numpy as np 
import tensorflow as tf

def predict(interpreter, test_ds, input_details, output_details):
    y_preds = []
    y_tests = []

    for x, y in test_ds:
        # give the input
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        # make prediction
        y_preds = interpreter.get_tensor(output_details[0]["index"])[0]
        y_tests.append(y.numpy()[0])
        
    return y_tests, y_preds
        
def get_mae(y_preds, y_tests):
    return np.abs(y_preds - y_tests)


def main(args):

    tensor_specs = (tf.TensorSpec([None, 6, 2], dtype=tf.float32), 
                tf.TensorSpec([None, 2]))
    
    test_ds = tf.data.experimental.load(f'./th_test', tensor_specs)
    test_ds = test_ds.unbatch().batch(1)

    interpreter = tf.lite.Interpreter(model_path='model_cnn_2.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_tests, y_preds = predict(interpreter, test_ds, input_details, output_details)

    maes = get_mae(y_tests, y_preds)

    print(f'Temperature\'s MAE = {maes[0][0]}')
    print(f'Humidity\'s MAE = {maes[0][1]}')
    

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='cnn')
    parser.add_argument('-l','--labels', type=int, default=2)

    args = parser.parse_args()

    main(args)