# RETINA
This project is aimed to act as an interactive assistant to humans to achieve tasks. There is a speech conversation with the system to command the system to detect objects in a live video feed. 

The user issues the wake up command to pass object name as speech and the systems returns a bounding box and speech feedback regarding the presence of the object.

### Custom Training

Apart from using the COCO dataset model, we also have a trained model to detect objects such as coat,  coin, calculator, dinosaur, seat belt, missile, pen, handgun, sunglasses, mug, watch. 

The weights for this model has not been uploaded due to it's size.

### Execution Instructions

- python3 launch.py --model 1

The above command will launch the application and accept speech input to detect objects in the COCO dataset. Custom dataset can be used instead by passing the argument as '2' to the model. eg. 

-  python3 launch.py --model 1 --voice_cmd False --currentclass 'cell phone'

The above command can be used to turn off the voice interface and detect objects using a pre-defined configuration.

### File Descriptions

- launch.py: Launches threads for video and speech to run in parallel. 
- custom_yolo.py: Runs the module for custom trained dataset
- speech_to_text.py: Controls the speech input and feedback module

