#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"


#include "caffe/layers/gradient_loss.hpp"
#include "caffe/layers/st_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/concat_layer.hpp"


namespace caffe {


template <typename Dtype>
void convolve_hv(Blob<Dtype>& dst,Blob<Dtype>& src, Blob<Dtype> & filter, int horiz,int vert ){
  int group_ = src.shape(1);
  int kernel_dim_= filter.count(0);
  Blob<Dtype> col_buffer_;
  vector<int> col_buffer_shape_;
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_*group_);
  col_buffer_shape_.push_back(src.shape(2));
  col_buffer_shape_.push_back(src.shape(3));
  col_buffer_.Reshape(col_buffer_shape_);
  
  int bottom_dim_=src.count(1);
  int top_dim_=dst.count(1);

  int filter_order=(filter.shape(1)-1)/2;
  int col_offset_ = kernel_dim_*dst.shape(2)*dst.shape(3);
  int out_offset_ = dst.shape(2)*dst.shape(3);
  
if(horiz&&vert){
  for(int n=0; n< src.shape(0);++n){

  Blob<Dtype> temp;
  vector<int> temp_shape(3);
  temp_shape[0]=src.shape(1);
  temp_shape[1]=src.shape(2);
  temp_shape[2]=src.shape(3);
  temp.Reshape(temp_shape);
  int temp_offset_=dst.shape(2)*dst.shape(3);
  Dtype* temp_data= temp.mutable_gpu_data();
  im2col_gpu(src.gpu_data()+n*bottom_dim_,src.shape(1),src.shape(2),src.shape(3),filter.shape(0),filter.shape(1),0,filter_order,1,1,1,1,col_buffer_.mutable_gpu_data());
  for(int g=0;g<group_;++g){
     caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,1,dst.shape(2)*dst.shape(3),kernel_dim_,(Dtype)1.,filter.gpu_data(),col_buffer_.gpu_data()+col_offset_*g,(Dtype)0.,temp_data+g*temp_offset_);
     }
  im2col_gpu(temp_data,src.shape(1),src.shape(2),src.shape(3),filter.shape(1),filter.shape(0),filter_order,0,1,1,1,1,col_buffer_.mutable_gpu_data());
  for(int g=0;g<group_;++g){
     caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,1,dst.shape(2)*dst.shape(3),kernel_dim_,(Dtype)1.,filter.gpu_data(),col_buffer_.gpu_data()+col_offset_*g,(Dtype)0.,dst.mutable_gpu_data()+n*top_dim_+g*out_offset_);
     }
   }
}else if(horiz && !vert){
 for(int n=0; n< src.shape(0);++n){
 
  im2col_gpu(src.gpu_data()+n*bottom_dim_,src.shape(1),src.shape(2),src.shape(3),filter.shape(0),filter.shape(1),0,filter_order,1,1,1,1,col_buffer_.mutable_gpu_data());
  for(int g=0;g<group_;++g){
     caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,1,dst.shape(2)*dst.shape(3),kernel_dim_,(Dtype)1.,filter.gpu_data(),col_buffer_.gpu_data()+col_offset_*g,(Dtype)0.,dst.mutable_gpu_data()+n*top_dim_+g*out_offset_);
   }
  }
}else if(!horiz&&vert){
  for(int n=0;n< src.shape(0);++n){
 
  im2col_gpu(src.gpu_data()+n*bottom_dim_,src.shape(1),src.shape(2),src.shape(3),filter.shape(1),filter.shape(0),filter_order,0,1,1,1,1,col_buffer_.mutable_gpu_data());
  for(int g=0;g<group_;++g){
     caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,1,dst.shape(2)*dst.shape(3),kernel_dim_,(Dtype)1.,filter.gpu_data(),col_buffer_.gpu_data()+col_offset_*g,(Dtype)0.,dst.mutable_gpu_data()+n*top_dim_+g*out_offset_);

   }
  }
 }
}




template <typename Dtype>
__global__ void ComputeSign(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? Dtype(1) : Dtype(-1);
  }
} 

template <typename Dtype>
__global__ void FindNotNaNs(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index]==in[index] ? Dtype(1) : Dtype(0);
  }
} 

template <typename Dtype>
__global__ void KillNaNs(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index]==in[index] ? in[index] : Dtype(0);
  }
}

template <typename Dtype>
__global__ void KillMasked(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > Dtype(0.5) ? out[index] : Dtype(0);
//     out[index] = out[index]==out[index] ? out[index] : Dtype(0);
//     out[index] = out[index]>1e3 ? 0 : out[index];
//     out[index] = out[index]<-1e3 ? 0 : out[index];
  }
}

template <typename Dtype>
__global__ void KillMaskedAcrossChannels(const int n, const int width_height, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int mask_idx = index % width_height;
    out[index] = in[mask_idx] > Dtype(0.5) ? out[index] : Dtype(0);
  }
}

template <typename Dtype>
__global__ void MaskPlateauValues(const int n, const Dtype* in, Dtype* out, Dtype plateau) {
  CUDA_KERNEL_LOOP(index, n) {
    if(fabs(in[index]) < plateau) out[index] = Dtype(0); // Mask out plateau values and keep other as is
  }
} 

template <typename Dtype>
__global__ void MaskPlateauValuesInitial(const int n, const Dtype* in, Dtype* out, Dtype plateau) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = (fabs(in[index]) < plateau) ? Dtype(0) : Dtype(1);
  }
} 


template <typename Dtype>
void GradientLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    
    convolve_hv<Dtype>(img1_gradx_,*bottom[0],image_filter,1,0);
    convolve_hv<Dtype>(img1_grady_,*bottom[0],image_filter,0,1);    

    convolve_hv<Dtype>(img2_gradx_,*bottom[1],image_filter,1,0);
    convolve_hv<Dtype>(img2_grady_,*bottom[1],image_filter,0,1);   

    concat1_layer_->Forward(concat1_bottom_vec_,concat1_top_vec_);
    concat2_layer_->Forward(concat2_bottom_vec_,concat2_top_vec_);

    stn_layer_->Forward(stn_bottom_vec_,stn_top_vec_);

    
  
    Dtype dot, loss;
  
    diff_layer_->Forward(diff_bottom_vec_, diff_top_vec_);
  
    Blob<Dtype> *diffptr = diff_top_vec_[0];
  
  // if necessary, compute the number of not-NaNs
  int count = bottom[0]->count()*2;
  int num = bottom[0]->num();
  FindNotNaNs<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, diffptr->gpu_data(), mask_.mutable_gpu_data());
  cudaDeviceSynchronize();
  CUDA_POST_KERNEL_CHECK;
  
  if (this->layer_param_.gradient_loss_param().normalize_by_num_entries()) {    
    caffe_gpu_dot(count, mask_.gpu_data(), mask_.gpu_data(), &normalize_coeff_);

    normalize_coeff_ /= mask_.channels();
  } else {
    normalize_coeff_ = num;
  }
  
  
    // set masked (NaNs only) to zero
    KillMasked<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, mask_.gpu_data(), diffptr->mutable_gpu_data());
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
    
    square_layer_->Forward(diff_top_vec_, square_top_vec_);
    sum_layer_->Forward(square_top_vec_, sum_top_vec_);
    
    // Mask plateau in summed blob (only one channel):
    if(this->layer_param_.gradient_loss_param().plateau() > 0) {
      float plateau_val_squared = this->layer_param_.gradient_loss_param().plateau() * this->layer_param_.gradient_loss_param().plateau();
      MaskPlateauValuesInitial<Dtype><<<CAFFE_GET_BLOCKS(sum_output_.count()), CAFFE_CUDA_NUM_THREADS>>>(
          sum_output_.count(), sum_output_.gpu_data(), plateau_l2_.mutable_gpu_data(), plateau_val_squared);
      cudaDeviceSynchronize();
      CUDA_POST_KERNEL_CHECK;
      
      KillMasked<Dtype><<<CAFFE_GET_BLOCKS(sum_output_.count()), CAFFE_CUDA_NUM_THREADS>>>(
            sum_output_.count(), plateau_l2_.gpu_data(), sum_output_.mutable_gpu_data());
      cudaDeviceSynchronize();
      CUDA_POST_KERNEL_CHECK;
    }
    
    sqrt_layer_->Forward(sum_top_vec_, sqrt_top_vec_);
    // Note sign_ is set to all ones in Reshape
    caffe_gpu_dot(sqrt_output_.count(), sqrt_output_.gpu_data(), sign_.gpu_data(), &dot);
  

    loss = dot / normalize_coeff_; 
    top[0]->mutable_cpu_data()[0] = loss;

 
}

template <typename Dtype>
void GradientLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{  
  bool prop_down = propagate_down[0];
  if(bottom.size() > 1) {prop_down |= propagate_down[1];
                         prop_down |= propagate_down[2];}
                        
  
  Blob<Dtype> *diffptr = diff_top_vec_[0];
  
  if (prop_down) {
    const Dtype alpha = top[0]->cpu_diff()[0] ;
    
      vector<bool> prop_down(1,true);
      caffe_gpu_axpby(sqrt_output_.count(), alpha, sign_.gpu_data(),                   
          Dtype(0), sqrt_output_.mutable_gpu_diff());
      sqrt_layer_->Backward(sqrt_top_vec_, prop_down, sum_top_vec_);
      
      if(this->layer_param_.gradient_loss_param().plateau() > 0) {
        KillMasked<Dtype><<<CAFFE_GET_BLOCKS(sum_output_.count()), CAFFE_CUDA_NUM_THREADS>>>(
              sum_output_.count(), plateau_l2_.gpu_data(), sum_output_.mutable_gpu_diff());
        cudaDeviceSynchronize();
        CUDA_POST_KERNEL_CHECK;
      }
      
      sum_layer_->Backward(sum_top_vec_, prop_down, square_top_vec_);
      square_layer_->Backward(square_top_vec_, prop_down, diff_top_vec_);
      
   
    KillMasked<Dtype><<<CAFFE_GET_BLOCKS(diffptr->count()), CAFFE_CUDA_NUM_THREADS>>>(
        diffptr->count(), mask_.gpu_data(), diffptr->mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
    
    vector<bool> propagate_down2(2,true);

       diff_layer_->Backward(diff_top_vec_, propagate_down2, diff_bottom_vec_);
       stn_layer_->Backward(stn_top_vec_,propagate_down2,stn_bottom_vec_);
    
  }
  
}

INSTANTIATE_LAYER_GPU_FUNCS(GradientLossLayer);

}  // namespace caffe
