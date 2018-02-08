#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/st_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void STLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	LossLayer<Dtype>::LayerSetUp(bottom, top);

	string prefix = "\t\tST Loss Layer:: LayerSetUp: \t";

	std::cout<<prefix<<"Getting output_H_ and output_W_"<<std::endl;

	output_H_ = bottom[0]->shape(2);
	if(this->layer_param_.st_loss_param().has_output_h()) {
		output_H_ = this->layer_param_.st_param().output_h();
	}
	output_W_ = bottom[0]->shape(3);
	if(this->layer_param_.st_loss_param().has_output_w()) {
		output_W_ = this->layer_param_.st_param().output_w();
	}


	std::cout<<prefix<<"output_H_ = "<<output_H_<<", output_W_ = "<<output_W_<<std::endl;
}

template <typename Dtype>
void STLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	vector<int> tot_loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(tot_loss_shape);
        
	N = bottom[0]->shape(0);
        C = bottom[0]->shape(1);
	vector<int> loss_shape(3);
	loss_shape[0] = N;
	loss_shape[1] = output_H_;
	loss_shape[2] = output_W_;
	loss_.Reshape(loss_shape);

        mask_.Reshape(loss_shape);

	
}

template <typename Dtype>
void STLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	/* not implemented */
	CHECK(false) << "Error: not implemented.";
}

template <typename Dtype>
void STLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	/* not implemented */
	CHECK(false) << "Error: not implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(STLossLayer);
#endif

INSTANTIATE_CLASS(STLossLayer);
REGISTER_LAYER_CLASS(STLoss);

}  // namespace caffe
