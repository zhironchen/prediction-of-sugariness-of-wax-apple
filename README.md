# prediction-of-sugariness-of-wax-apple-code
The codes of predicting sugariness of wax-apple using HSIs.

Workflow:
1. Data preparation & visualization: 
    save_pkl.py : For transforming .mat file to .pkl file, this step will obtain the raw HSIs data
    data_cropping_and_sampling.py : For obtaining training, validation and test data, random_sample.py and roi.py will be used in this step.
    visualization_input_data.py For visualizaing all the data for training.
2. Modelling:
    regression.py : To construct deep learning models and train the models, set "mode" to "test" after model training.
3. Visualization of the outputs of the layer before the last output layer:
    get_layer_output.py : To obtain the outputs of the layer before the last output layer.
    visualization_layer_output.py : Visualizing the outputs.