// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

#include "caffe/layers/mean_layer.hpp"


using std::max;

namespace caffe {
  
template <typename Dtype>
void MeanLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{


     scale=this->layer_param().mean_param().scale();

     vector<int> shape(1);
     mean_num=this->layer_param_.mean_param().value().size();
     shape[0]=mean_num;
     mean_.Reshape(shape);
     for(int i=0;i<mean_num;i++)
          mean_.mutable_cpu_data()[i]=this->layer_param_.mean_param().value().Get(i);
      

}


template <typename Dtype>
void MeanLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    CHECK_GT(top.size(), 0) << "MEAN needs one output";
    CHECK_EQ(top.size(), bottom.size()) << "MEAN needs as many outputs as inputs";

    
    
        for(int k=0; k<bottom.size(); k++)
            CHECK_EQ(bottom[k]->channels() % this->layer_param_.mean_param().value().size(), 0) << "bottom size not divisible by mean size";
    

    for(int k=0; k<bottom.size(); k++)
        top[k]->Reshape(bottom[k]->num(),bottom[k]->channels(),bottom[k]->height(),bottom[k]->width());
}

template <typename Dtype>
void MeanLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    	/* not implemented */
	CHECK(false) << "Error: not implemented.";
}





INSTANTIATE_CLASS(MeanLayer);
REGISTER_LAYER_CLASS(Mean);


}  // namespace caffe
