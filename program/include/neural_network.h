#ifndef LICENSE_PLATE_RECOGNITION_NEURAL_NETWORK_H
#define LICENSE_PLATE_RECOGNITION_NEURAL_NETWORK_H

#include <Python.h>

struct neural_network {
    const char classes[34] = {
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    };

    neural_network() {
        Py_Initialize();
        PyRun_SimpleString("from keras.models import load_model");
        PyRun_SimpleString("from keras.preprocessing.image import load_img, img_to_array");
        PyRun_SimpleString("import numpy as np");
        PyRun_SimpleString("model = load_model('nn_model.h5')");
    }

    ~neural_network() {
        Py_Finalize();
    }

    char predict() {
        PyObject* dict = PyDict_New();
        PyRun_String("result = model.predict_classes(image)[0]", Py_file_input, PyModule_GetDict(PyImport_AddModule("__main__")), dict);
        return classes[PyLong_AsLong(PyDict_GetItemString(dict, "result"))];
    }

    void set_image(Mat img) {
        std::string pystring = "image = ";
        resize(img, img, Size(20, 20), INTER_NEAREST);

        pystring += "[";
        for (int i = 0; i < img.rows; i++) {
            if (i > 0) pystring += ",";
            pystring += "[";
            for (int j = 0; j < img.cols; j++) {
                if (j > 0) pystring += ",";
                pystring += "[";
                pystring += std::to_string(img.at<uchar>(i, j));
                pystring += "]";
            }
            pystring += "]";
        }
        pystring += "]";

        PyRun_SimpleString(pystring.c_str());
        PyRun_SimpleString("image = np.expand_dims(image, axis = 0)");
    }
};

#endif
