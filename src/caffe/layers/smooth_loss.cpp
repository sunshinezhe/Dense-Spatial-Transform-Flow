
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/smooth_loss.hpp"
#include "caffe/layers/power_layer.hpp"


namespace caffe {


template <typename Dtype>
void filter_filler(float*  half_coeffs,Blob<Dtype>& filter,const int order){

   vector<int> shape(2);
   shape[0]=1;
   shape[1]=2*order+1;
   filter.Reshape(shape);
   Dtype* filter_data=filter.mutable_cpu_data();
   int  i;
   for(i=0;i<=order;i++){
       filter_data[order-i]=+half_coeffs[i];
       filter_data[order+i]=-half_coeffs[i]; 
   }

}

template <typename Dtype>
void SmoothLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);  
  
  
    N=bottom[0]->shape(0);
    C=bottom[0]->shape(1);
    H=bottom[0]->shape(2);
    W=bottom[0]->shape(3);

 
    horiz_loss_.Reshape(N,1,H,W);
    verti_loss_.Reshape(N,1,H,W);
    
   
    // Set up power layer to compute elementwise sqrt
    sqrt1_bottom_vec_.clear();
    sqrt1_bottom_vec_.push_back(&horiz_loss_);
    sqrt1_top_vec_.clear();
    sqrt1_top_vec_.push_back(&sqrt1_output_);
    LayerParameter sqrt1_param;
    sqrt1_param.mutable_power_param()->set_power(0.5);
    sqrt1_param.mutable_power_param()->set_shift(this->layer_param_.smooth_loss_param().epsilon());
    sqrt1_layer_.reset(new PowerLayer<Dtype>(sqrt1_param));
    sqrt1_layer_->SetUp(sqrt1_bottom_vec_, sqrt1_top_vec_);

    sqrt2_bottom_vec_.clear();
    sqrt2_bottom_vec_.push_back(&verti_loss_);
    sqrt2_top_vec_.clear();
    sqrt2_top_vec_.push_back(&sqrt2_output_);
    LayerParameter sqrt2_param;
    sqrt2_param.mutable_power_param()->set_power(0.5);
    sqrt2_param.mutable_power_param()->set_shift(this->layer_param_.smooth_loss_param().epsilon());
    sqrt2_layer_.reset(new PowerLayer<Dtype>(sqrt2_param));
    sqrt2_layer_->SetUp(sqrt2_bottom_vec_, sqrt2_top_vec_);

  float image_hcoeffs[3]={0.0f,-8.0f/12.0f,1.0f/12.0f};
  
  filter_filler<Dtype>(image_hcoeffs,image_filter,2);
  
}

template <typename Dtype>
void SmoothLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  
  N=bottom[0]->shape(0);
  C=bottom[0]->shape(1);
  H=bottom[0]->shape(2);
  W=bottom[0]->shape(3);

  gx1_.Reshape(N,C,H,W);
  gy1_.Reshape(N,C,H,W);
  gx2_.Reshape(N,C,H,W);
  gy2_.Reshape(N,C,H,W);

  lum.Reshape(N,1,H,W);
  lum_x_.Reshape(N,1,H,W);
  lum_y_.Reshape(N,1,H,W);
 
  horiz_loss_.Reshape(N,1,H,W);
  verti_loss_.Reshape(N,1,H,W);
  d2_tmp_.Reshape(N,4,H*W,2);
  

  smoothweight_.Reshape(N,1,H,W);
  smoothweight_h_.Reshape(N,1,H,W);
  smoothweight_v_.Reshape(N,1,H,W);


  plateau1_l2_.ReshapeLike(horiz_loss_);
  plateau2_l2_.ReshapeLike(verti_loss_);
  
  sign_.Reshape(bottom[0]->num(), bottom[0]->channels(),
               bottom[0]->height(), bottom[0]->width());
 
  sqrt1_layer_->Reshape(sqrt1_bottom_vec_, sqrt1_top_vec_);
  sqrt2_layer_->Reshape(sqrt2_bottom_vec_, sqrt2_top_vec_);    
 
  caffe_set(sign_.count()/sign_.channels(), Dtype(1), sign_.mutable_cpu_data());
  
  
}






template <typename Dtype>
void SmoothLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SmoothLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SmoothLossLayer);
#endif

INSTANTIATE_CLASS(SmoothLossLayer);
REGISTER_LAYER_CLASS(SmoothLoss);

}  // namespace caffe
