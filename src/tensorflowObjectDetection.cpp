//
// Created by jonas on 11/6/17.
//

#include <tiff.h>

#include <utility>
#include "iCub/tensorflowObjectDetection.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::TensorShape;



inline string replaceChar(string str, char ch1, char ch2) {
    for (int i = 0; i < str.length(); ++i) {
        char tmp = str[i];
        if (str[i] == ch1)
            str[i] = ch2;
    }

    return str;
}


tensorflowObjectDetection::tensorflowObjectDetection(std::string t_pathGraph, std::string t_pathLabels,  std::string t_model_name) {

    this->m_pathToGraph = std::move(t_pathGraph);
    this->m_pathToLabels = std::move(t_pathLabels);

    this->m_input_layer = "image_tensor:0";
    this->m_output_layer =  { "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0" };
    this->m_model_name = std::move(t_model_name);

    this->m_detectionThreshold = 0.5;

    size_t label_count;
    Status read_labels_status = ReadLabelsFile(m_pathToLabels, &m_labels, &label_count);


    if (!read_labels_status.ok()) {
        LOG(ERROR) << read_labels_status;
    }



}


/************************************* CORE FUNCTIONS  *************************************/





Status tensorflowObjectDetection::ReadLabelsFile(const string &file_name,  std::map<int, std::string> *result,
                                           size_t *found_label_count) {
    std::ifstream file(file_name);
    if (!file) {
        return tensorflow::errors::NotFound("Labels file ", file_name,
                                            " not found.");
    }
    result->clear();
    string line;
    int id;
    while (std::getline(file, line)) {
        if(line.find("id:") != std::string::npos){

            std::size_t pos = line.find(':');
            const char*  tmp = line.substr(pos+1).c_str();
            id = atoi(tmp);
        }


        if(line.find("display_name") != std::string::npos){

            std::size_t pos = line.find(':');
            std::string className = line.substr(pos+3);
            className = replaceChar(className, '"', ' ');
            result->insert(std::pair<int,string>(id, className));


        }
    }

    return Status::OK();
}

tensorflow::Tensor tensorflowObjectDetection::MatToTensor(cv::Mat inputImage) {

    const int inputImageHeight =  inputImage.rows;
    const int inputImageWidth =  inputImage.cols;
    auto root = tensorflow::Scope::NewRootScope();

    // allocate a Tensor
    Tensor tensorImage(tensorflow::DT_UINT8, TensorShape({1, inputImageHeight, inputImageWidth, 3}));



    // get pointer to memory for that Tensor
    auto *p = tensorImage.flat<uint8>().data();
    // create a "fake" cv::Mat from it
    cv::Mat matToTensor(inputImageHeight, inputImageWidth, CV_8U, p);



    // use it here as a destination
    inputImage.convertTo(matToTensor, CV_8U);

    //normalizeTensor(&tensorImage);

    return tensorImage;

}

tensorflow::Status tensorflowObjectDetection::LoadGraph(const std::string &graph_file_name, std::unique_ptr<tensorflow::Session> *session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

tensorflow::Status tensorflowObjectDetection::PrintTopLabels(std::vector<tensorflow::Tensor> &outputs,
                                                       const std::string &labels_file_name) {



    auto boxes = outputs[0].flat_outer_dims<float,3>();
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();

    LOG(ERROR) << "number of detection:" << num_detections << std::endl;

    m_objectsDetected.clear();
    for(size_t i = 0; i < num_detections(0) && i < 20;++i)
    {
        if(scores(i) > m_detectionThreshold)
        {
            int boxRectangleX1 = m_widthInputImage*boxes(0,i,1);
            int boxRectangleY1 = m_heightInputImage*boxes(0,i,0);

            int boxRectangleX2 = m_widthInputImage*boxes(0,i,3);
            int boxRectangleY2 = m_heightInputImage*boxes(0,i,2);

            const Box boxCoordinates = {{boxRectangleX1, boxRectangleY1, boxRectangleX2, boxRectangleY2}, scores(i)};
            const string labelName = m_labels[classes(i)];


            m_objectsDetected.insert(std::pair<string, Box>( labelName, boxCoordinates ));


            LOG(INFO) << i << ",score:" << scores(i)<< ",classID:" << classes(i) << ", "<< ",class:" << m_labels[classes(i)] << ",box:" << "," << boxRectangleX1 << "," << boxRectangleY1 << "," << boxRectangleX2 << "," << boxRectangleY2;

        }
    }


    return Status::OK();
}



std::string  tensorflowObjectDetection::getDetectedObjectToString() {

    string objectsDetected;
    for (auto &it : m_objectsDetected) {
        objectsDetected.append(it.first + " : " + boxToString(it.second) +" ; ");
    }

        return objectsDetected;
}




std::string tensorflowObjectDetection::inferObject(cv::Mat t_inputImage) {

    this->m_inputImage = t_inputImage;
    this->m_widthInputImage = t_inputImage.cols;
    this->m_heightInputImage = t_inputImage.rows;

    const Tensor resized_tensor = MatToTensor(t_inputImage);
    std::vector<Tensor> outputs;


    Status run_status = m_session->Run({{m_input_layer, resized_tensor}},
                                     {m_output_layer}, {}, &outputs);

    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return "";
    } else {
        PrintTopLabels(outputs, this->m_pathToLabels);

        return getDetectedObjectToString();

    }



}

tensorflow::Status tensorflowObjectDetection::initGraph() {

    if(!initPreprocessParameters(m_model_name)){
        return Status(tensorflow::error::FAILED_PRECONDITION, "Unable to compute the model architecture, check the model_name parameters");

    }

    Status load_graph_status = LoadGraph(this->m_pathToGraph, &m_session);

    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        return Status(tensorflow::error::FAILED_PRECONDITION,"Unable to initialize the graph, check the graph and labels path");
    }


    return Status::OK();

}

bool tensorflowObjectDetection::initPreprocessParameters(std::string modelName) {

    return true;
}


double tensorflowObjectDetection::getM_detecttionThreshold() const {
    return m_detectionThreshold;
}

void tensorflowObjectDetection::setM_detectionThreshold(double m_inferencethreshold) {
    tensorflowObjectDetection::m_detectionThreshold = m_inferencethreshold;
}




