#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/dynamic_bilinear.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DynamicBilinearLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

		

	std::cout<<"Getting output_H_ and output_W_"<<std::endl;

       if(this->layer_param_.dynamic_bilinear_param().has_output_h()&&this->layer_param_.dynamic_bilinear_param().has_output_w()){
 
              output_H_ = this->layer_param_.dynamic_bilinear_param().output_h();
              output_W_ = this->layer_param_.dynamic_bilinear_param().output_w();
        }else if(this->layer_param_.dynamic_bilinear_param().has_scale()){

              output_H_ =bottom[0]->shape(2)*this->layer_param_.dynamic_bilinear_param().scale();
              output_W_ =bottom[0]->shape(3)*this->layer_param_.dynamic_bilinear_param().scale();
             
        }else if(bottom.size()>1){
              
              output_H_ =bottom[1]->shape(2);
              output_W_ =bottom[1]->shape(3);
        }else{
        
              CHECK(false) << "No output size specified ";

        }

	std::cout<<"output_H_ = "<<output_H_<<", output_W_ = "<<output_W_<<std::endl;

      
	std::cout<<"Initialization finished."<<std::endl;
}

template <typename Dtype>
void DynamicBilinearLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tDynamic Bilinear Layer:: Reshape: \t";

	if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

	N = bottom[0]->shape(0);
	C = bottom[0]->shape(1);
	H = bottom[0]->shape(2);
	W = bottom[0]->shape(3);

	// reshape V
	vector<int> shape(4);

	shape[0] = N;
	shape[1] = C;
	shape[2] = output_H_;
	shape[3] = output_W_;

	top[0]->Reshape(shape);



	if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}



template <typename Dtype>
void DynamicBilinearLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	/* not implemented */
	CHECK(false) << "Error: not implemented.";
}



		


template <typename Dtype>
void DynamicBilinearLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

		/* not implemented */
	CHECK(false) << "Error: not implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(DynamicBilinearLayer);
#endif

INSTANTIATE_CLASS(DynamicBilinearLayer);
REGISTER_LAYER_CLASS(DynamicBilinear);

}  // namespace caffe
