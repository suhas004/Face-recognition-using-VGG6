import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class cricket:
    def __init__(self,filename):
        self.filename =filename


    def prediction_cricket(self):
        # load model
        model = load_model('resnet.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        print(result)
        if result[0][0] >=0.45:
            prediction = 'dhoni'
            print("dhoni")

        elif result[0][1] > 0.45:
            print('kohli')



        if result[0][2] >0.45:
            prediction = 'sachin'
            print("sachin")


file='kohili.jpg'

a=cricket(file)
a.prediction_cricket()

