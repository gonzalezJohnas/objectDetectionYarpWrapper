//
// Created by jonas on 11/6/17.
//

#ifndef OBJECTRECOGNITIONINFER_tensorflowInference_H
#define OBJECTRECOGNITIONINFER_tensorflowInference_H

// Standard import
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

// Tensorflow import
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/lib/core/stringpiece.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/util/command_line_flags.h>

// OpenCV import
#include <opencv2/core/mat.hpp>
#include <opencv/cv.hpp>

class tensorflowInference {
public:

    /**
     * Defautls constructor
     * @param pathGraph
     * @param pathLabels
     */
    tensorflowInference(std::string pathGraph, std::string pathLabels, std::string model_name);


    std::string inferObject(cv::Mat t_inputImage);


    /**
     * Initialize the networks by loading the graph and labels
     * @return Tensorflow::Status
     */
    tensorflow::Status initGraph();

    /**
     * Get the boxes image return by the DeepNetwork
     * @return
     */
    const cv::Mat &getM_outputBoxesImage() const;

    /**
     * Get the current threshold for inference to consider valid classification
     * @return
     */
    double getM_inferencethreshold() const;

    /**
     * Set the threshold for inference to consider valid classification
     * @param m_inferencethreshold
     */
    void setM_inferencethreshold(double m_inferencethreshold);

private:
    // Parameters of the Deepnetworks graph
    std::unique_ptr<tensorflow::Session> m_session;
    std::string m_pathToGraph;
    std::string m_pathToLabels;
    std::string m_input_layer;
    std::string m_model_name;


    // Parameters for the output
    std::vector<std::string> m_output_layer;
    std::map<int,std::string> m_labels;

    // Parameters for the Image input and Output
    cv::Mat m_inputImage;
    cv::Mat m_outputBoxesImage;

    int m_withInputImage;
    int m_heightInputImage;

    // Parameters for the Inference
    double m_inferencethreshold;


    /**
     * Takes a file name, and loads a list of labels from it, one per line, and
     * returns a vector of the strings. It pads with empty strings so the length
     * of the result is a multiple of 16, because our model expects that.
     * @param file_name
     * @param result
     * @param found_label_count
     * @return Tensor status of the success of the process
     */
    tensorflow::Status ReadLabelsFile(const std::string &file_name, std::map<int, std::string> *result,
                                      size_t *found_label_count);

    /**
     * Convert a Mat OpenCV object into a Tensor
     * @param inputHeight
     * @param inputWidth
     * @param inputImage
     * @return Tensor Object
     */
    tensorflow::Tensor MatToTensor(cv::Mat inputImage);



    /**
     * Reads a model graph definition from disk, and creates a session object you can use to run it.
     * @param graph_file_name
     * @param session
     * @return Tensor status of the success of the process
     */
    tensorflow::Status LoadGraph(const std::string &graph_file_name,
                                 std::unique_ptr<tensorflow::Session> *session);



    /**
     * Given the output of a model run, and the name of a file containing the labels
     * this prints out the top five highest-scoring values.
     * @param outputs
     * @param labels_file_name
     * @return Tensor status of the success of the process
     */
    tensorflow::Status PrintTopLabels(std::vector<tensorflow::Tensor> &outputs,
                          const std::string &labels_file_name);

    /**
     * Given the output of a model run, and the name of a file containing the labels
     * this return the top class
     * @param outputs
     * @param labels_file_name
     * @return Tensor status of the success of the process
     */
    std::string GetTopClass(const std::vector<tensorflow::Tensor> &outputs,
                                      const std::string &labels_file_name);


    /**
     * This functions set the right parameters according to the graph used ( InceptionV3, MobileNet...) :
     * - Mean
     * - Std
     * - Input images dimensions
     * @param modelName
     */
    bool initPreprocessParameters(std::string modelName);;
};


#endif //OBJECTRECOGNITIONINFER_tensorflowInference_H
