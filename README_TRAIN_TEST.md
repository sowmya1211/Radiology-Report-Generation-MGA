## Training

The full model is trained in 3 training stages:

1. Object detector
2. Object detector + Region Selection Classifier model - full_model without LM
3. Full model (alongwith Report Generation Model) - full_model with LM

### Object detector

For training the object detector, specify the training configurations (e.g. batch size etc.) in  [training_script_object_detector.py](src/object_detector/training_script_object_detector.py) then run 
"**python  [training_script_object_detector.py](src/object_detector/training_script_object_detector.py)**".

The weights of the trained object detector model will be stored in the folder specified in [path_datasets_and_weights.py](src/path_datasets_and_weights.py)

### Object detector +  + Region Selection Classifier model

For the second training stage, first specify the path to the best trained object detector in  [report_generation_model.py](src/full_model/report_generation_model.py), such that the trained object detector will be trained together with the 2 binary classifiers.
Next, specify the run configurations in [run_configurations.py](src/full_model/run_configurations.py). In particular, set "**PRETRAIN_WITHOUT_LM_MODEL = True**",
such that the language model is fully excluded from training. 
Start training by running "**python  [train_full_model.py](src/full_model/train_full_model.py)**".

### Full model
For the third training stage, adjust the run configurations in [run_configurations.py](src/full_model/run_configurations.py). In particular, set "**PRETRAIN_WITHOUT_LM_MODEL = False**", and set the batch size to small value, since the full model requires a lot of memory.
Next, specify the checkpoint of the best pre-trained model of training stage 2 in the main function of [train_full_model.py](src/full_model/train_full_model.py).

## Testing
Specify the run and checkpoint of the best trained full model to be tested in [test_set_evaluation.py](src/full_model/test_set_evaluation.py). 
Then run "**python [test_set_evaluation.py](src/full_model/test_set_evaluation.py)**". Text files with the test set scores (and generated reports/sentences) will be saved.