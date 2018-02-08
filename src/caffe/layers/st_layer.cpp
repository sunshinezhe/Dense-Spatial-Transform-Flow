#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/st_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tSpatial Transformer Layer:: LayerSetUp: \t";

	if(this->layer_param_.st_param().transform_type() == "Dense") {
		transform_type_ = "Dense";
	} else {
		CHECK(false) << prefix << "Transformation type only supports Dense now!" << std::endl;
	}

	if(this->layer_param_.st_param().sampler_type() == "bilinear") {
		sampler_type_ = "bilinear";
	} else {
		CHECK(false) << prefix << "Sampler type only supports bilinear now!" << std::endl;
	}

	if(this->layer_param_.st_param().to_compute_du()) {
		to_compute_dU_ = true;
	}

	std::cout<<prefix<<"Getting output_H_ and output_W_"<<std::endl;

	output_H_ = bottom[0]->shape(2);
	if(this->layer_param_.st_param().has_output_h()) {
		output_H_ = this->layer_param_.st_param().output_h();
	}
	output_W_ = bottom[0]->shape(3);
	if(this->layer_param_.st_param().has_output_w()) {
		output_W_ = this->layer_param_.st_param().output_w();
	}

	std::cout<<prefix<<"output_H_ = "<<output_H_<<", output_W_ = "<<output_W_<<std::endl;

	/*std::cout<<prefix<<"Getting pre-defined parameters"<<std::endl;

	is_pre_defined_theta[0] = false;
	if(this->layer_param_.st_param().has_theta_1_1()) {
		is_pre_defined_theta[0] = true;
		++ pre_defined_count;
		pre_defined_theta[0] = this->layer_param_.st_param().theta_1_1();
		std::cout<<prefix<<"Getting pre-defined theta[1][1] = "<<pre_defined_theta[0]<<std::endl;
	}

	is_pre_defined_theta[1] = false;
	if(this->layer_param_.st_param().has_theta_1_2()) {
		is_pre_defined_theta[1] = true;
		++ pre_defined_count;
		pre_defined_theta[1] = this->layer_param_.st_param().theta_1_2();
		std::cout<<prefix<<"Getting pre-defined theta[1][2] = "<<pre_defined_theta[1]<<std::endl;
	}

	is_pre_defined_theta[2] = false;
	if(this->layer_param_.st_param().has_theta_1_3()) {
		is_pre_defined_theta[2] = true;
		++ pre_defined_count;
		pre_defined_theta[2] = this->layer_param_.st_param().theta_1_3();
		std::cout<<prefix<<"Getting pre-defined theta[1][3] = "<<pre_defined_theta[2]<<std::endl;
	}

	is_pre_defined_theta[3] = false;
	if(this->layer_param_.st_param().has_theta_2_1()) {
		is_pre_defined_theta[3] = true;
		++ pre_defined_count;
		pre_defined_theta[3] = this->layer_param_.st_param().theta_2_1();
		std::cout<<prefix<<"Getting pre-defined theta[2][1] = "<<pre_defined_theta[3]<<std::endl;
	}

	is_pre_defined_theta[4] = false;
	if(this->layer_param_.st_param().has_theta_2_2()) {
		is_pre_defined_theta[4] = true;
		++ pre_defined_count;
		pre_defined_theta[4] = this->layer_param_.st_param().theta_2_2();
		std::cout<<prefix<<"Getting pre-defined theta[2][2] = "<<pre_defined_theta[4]<<std::endl;
	}

	is_pre_defined_theta[5] = false;
	if(this->layer_param_.st_param().has_theta_2_3()) {
		is_pre_defined_theta[5] = true;
		++ pre_defined_count;
		pre_defined_theta[5] = this->layer_param_.st_param().theta_2_3();
		std::cout<<prefix<<"Getting pre-defined theta[2][3] = "<<pre_defined_theta[5]<<std::endl;
	}  */

	// check the validation for the parameter flow
	
	CHECK(bottom[1]->shape(0) == bottom[0]->shape(0)) << "The first dimension of flow and " <<
			"U should be the same" << std::endl;

	/* // initialize the matrix for output grid
	std::cout<<prefix<<"Initializing the matrix for output grid"<<std::endl;

	vector<int> shape_output(2);
	shape_output[0] = output_H_ * output_W_; shape_output[1] = 3;
	output_grid.Reshape(shape_output);

	Dtype* data = output_grid.mutable_cpu_data();
	for(int i=0; i<output_H_ * output_W_; ++i) {
		data[3 * i] = (i / output_W_) * 1.0 / output_H_ * 2 - 1;
		data[3 * i + 1] = (i % output_W_) * 1.0 / output_W_ * 2 - 1;
		data[3 * i + 2] = 1;
	}    */

      /*	// initialize the matrix for input grid
	std::cout<<prefix<<"Initializing the matrix for input grid"<<std::endl;

	vector<int> shape_input(3);
	shape_input[0] = bottom[1]->shape(0); shape_input[1] = output_H_ * output_W_; shape_input[2] = 2;
	input_grid.Reshape(shape_input);
      */
	std::cout<<prefix<<"Initialization finished."<<std::endl;
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tSpatial Transformer Layer:: Reshape: \t";

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

	// reshape dTheta_tmp
	vector<int> dTheta_tmp_shape(4);

	dTheta_tmp_shape[0] = N;
	dTheta_tmp_shape[1] = 2;
	dTheta_tmp_shape[2] = output_H_ * output_W_;
	dTheta_tmp_shape[3] = C;

	dTheta_tmp.Reshape(dTheta_tmp_shape);
       

	// init all_ones_2
	vector<int> all_ones_2_shape(1);
	all_ones_2_shape[0] =  C;
	all_ones_2.Reshape(all_ones_2_shape);
        /*
	// reshape full_theta
	vector<int> full_theta_shape(2);
	full_theta_shape[0] = N;
	full_theta_shape[1] = 6;
	full_theta.Reshape(full_theta_shape);
         */

	if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
Dtype SpatialTransformerLayer<Dtype>::transform_forward_cpu(const Dtype* pic, Dtype px, Dtype py) {

	bool debug = false;

	string prefix = "\t\tSpatial Transformer Layer:: transform_forward_cpu: \t";

	if(debug) std::cout<<prefix<<"Starting!\t"<<std::endl;
	if(debug) std::cout<<prefix<<"(px, py) = ("<<px<<", "<<py<<")"<<std::endl;

	Dtype res = (Dtype)0.;

	Dtype x = (px + 1) / 2 * H; Dtype y = (py + 1) / 2 * W;

	if(debug) std::cout<<prefix<<"(x, y) = ("<<x<<", "<<y<<")"<<std::endl;

	int m, n; Dtype w;

	m = floor(x); n = floor(y); w = 0;
	if(debug) std::cout<<prefix<<"1: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	m = floor(x) + 1; n = floor(y); w = 0;
	if(debug) std::cout<<prefix<<"2: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	m = floor(x); n = floor(y) + 1; w = 0;
	if(debug) std::cout<<prefix<<"3: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	if(debug) std::cout<<prefix<<"4: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	if(debug) std::cout<<prefix<<"Finished. \tres = "<<res<<std::endl;

	return res;
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	/* not implemented */
	CHECK(false) << "Error: not implemented.";
}



		


template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

		/* not implemented */
	CHECK(false) << "Error: not implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(SpatialTransformerLayer);
#endif

INSTANTIATE_CLASS(SpatialTransformerLayer);
REGISTER_LAYER_CLASS(SpatialTransformer);

}  // namespace caffe
