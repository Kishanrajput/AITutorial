
#include "QuickTutorialProcessor.h"
#include <JANA/JLogger.h>
#include "../include/Model.h"
#include "../include/Tensor.h"
//#include "opencv2/opencv.hpp"
#include <algorithm>
#include <iterator>

#include <numeric>
#include <iomanip>
#include <fstream>
#include <string.h>

#include <bits/stdc++.h> 

QuickTutorialProcessor::QuickTutorialProcessor() {
    SetTypeName(NAME_OF_THIS); // Provide JANA with this class's name
}

void QuickTutorialProcessor::Init() {
    LOG << "QuickTutorialProcessor::Init" << LOG_END;
    // Open TFiles, set up TTree branches, etc
    
    // New code here --------------------------------
    //Model model("../model.pb");

    //start = clock();
    // Tensor input_a{model, "input_a"};
    // Tensor input_b{model, "input_b"};
    // Tensor output{model, "result"};
    
    // New code ends --------------------------------
}

void QuickTutorialProcessor::Process(const std::shared_ptr<const JEvent> &event) {
    LOG << "QuickTutorialProcessor::Process, Event #" << event->GetEventNumber() << LOG_END;
    
    /// Do everything we can in parallel
    /// Warning: We are only allowed to use local variables and `event` here
    //auto hits = event->Get<Hit>();

    /// Lock mutex
    // std::lock_guard<std::mutex>lock(m_mutex);

    /// Do the rest sequentially
    /// Now we are free to access shared state such as m_heatmap
    //for (const Hit* hit : hits) {
        /// Update shared state
    //}

    // Model model("../model.pb");
    // model.init();

    // Tensor input_a{model, "input_a"};
    // Tensor input_b{model, "input_b"};
    // Tensor output{model, "result"};

    // std::vector<float> data(100);
    // std::iota(data.begin(), data.end(), 0);

    // input_a.set_data(data);
    // input_b.set_data(data);

    // model.run({&input_a, &input_b}, output);
    // for (float f : output.get_data<float>()) {
    //     std::cout << f << " ";
    // }
    // std::cout << std::endl;

    // Create model
    std::cout<<"Loading Model..........."<<std::endl;
    Model m("../MNIST_new/model.pb");
    std::cout<<"Model Loaded.............."<<std::endl;

    m.restore("../MNIST_new/checkpoint/train.ckpt");
    std::cout<<"Weights loaded and initialized.........."<<std::endl;
    
    // Create Tensors
    Tensor input(m, "input", 1, 784);
    Tensor prediction(m, "prediction", 1, 10);

    std::cout<<"Tensors Created!!"<<std::endl;

    // Read image
    //for (int i=0; i<10; i++) {
    //    cv::Mat img, scaled;

        // Read image
        //img = cv::imread("../images/"+std::to_string(i)+".png");

        // Scale image to range 0-1
        //img.convertTo(scaled, CV_64F, 1.f/255);


    std::ifstream inFile;

    inFile.open("../Images_large.txt");

    if(!inFile) {
        std::cout << "Unable to open images file..." << std::endl;
    }
    char c;
    int count = 0;
    std::string line;
    while(getline(inFile, line)){
        //std::cout<<"Character: "<<c<<"  :"<<std::endl;
        std::vector<double> img_data;
        float num = 0;
        for(int i=2;i<line.length();i++)
        {
            if(line[i] != ',')
            {
                num = num * 10 + float(line[i]);
            }
            else
            {
                img_data.push_back(num/256);
                num = 0;
            }
        }

//        // Returns first token
//        char *token = strtok(line, ",");
//
//        // Keep printing tokens while one of the
//        // delimiters present in str[].
//        while (token != NULL)
//        {
//            //printf("%s\n", token);
//            img_data.push_back(float(token)/256);
//            token = strtok(NULL, ",");
//        }

        //img_data.assign(scaled.begin<double>(), scaled.end<double>());
        //std::cout<<"Shape of img_data: "<<img_data.size()<<std::endl;
        // Feed data to input tensor
        input.set_data(img_data);
        //std::cout<<"Shape of Input Tensor is "<<input.DebugString()<<std::endl;
        //std::cout<<"Now running inference!!"<<std::endl;

        //std::cout << "Image received, now inferring!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        // Run and show predictions
        m.run(input, prediction);

        //std::cout<<"Inference done!!"<<std::endl;
        // Get tensor with predictions
        auto result = prediction.Tensor::get_data<double>();

        //std::cout<<"Results prepared!!"<<std::endl;

        // Maximum prob
        auto max_result = std::max_element(result.begin(), result.end());

        // Print result
        std::cout << "Real label: " << line[0] << ", predicted: " << std::distance(result.begin(), max_result)
            << ", Probability: " << (*max_result) << std::endl;
        count = count + 1;

        // Put image in vector

    }
}

void QuickTutorialProcessor::Finish() {
    // Close any resources
    LOG << "QuickTutorialProcessor::Finish" << LOG_END;
    //double time_taken = double(clock() - start) / double(CLOCKS_PER_SEC); 
    //cout << "Time taken by program is : " << time_taken; 
    //cout << " sec " << endl; 
}

