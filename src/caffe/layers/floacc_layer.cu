#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/floacc_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {



template <typename Dtype>
__global__ void compute_error(const int threads, int H,int W, const Dtype* error_s, Dtype* error_){

  CUDA_KERNEL_LOOP(index, threads) {
   int  t=index % W;
   int  s=index/W % H;
   int  i=index/(W*H);

     Dtype error_x, error_y;

    error_x=error_s[i*(2*W*H)+s*W+t];
    error_y=error_s[i*(2*W*H)+W*H+s*W+t];
    error_[index]=sqrt(error_x*error_x+error_y*error_y);

 
  }
}


template <typename Dtype>
__global__ void FindNotNaNs(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index]==in[index] ? Dtype(1) : Dtype(0);
  }
} 



template <typename Dtype>
__global__ void KillMasked_acc(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > Dtype(0.5) ? out[index] : Dtype(3.39615e+38);
//     out[index] = out[index]==out[index] ? out[index] : Dtype(0);
//     out[index] = out[index]>1e3 ? 0 : out[index];
//     out[index] = out[index]<-1e3 ? 0 : out[index];
  }
}


template <typename Dtype>
__global__ void subtrack_error(const int count,int H, int W, const Dtype* pred_,const Dtype* groundth_ ,Dtype* error_s){

CUDA_KERNEL_LOOP(index, count) {

      error_s[index]=pred_[index]-groundth_[index];
  
  }
}

template <typename Dtype>
__global__ void Finderror(const int threads,int H,int W,const Dtype* error_, Dtype* sign_,float threshold){

CUDA_KERNEL_LOOP(index, threads) {

   sign_[index]=error_[index]<=threshold ? Dtype(1):Dtype(0);

  }
}




template <typename Dtype>
void FloaccLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
   int count = bottom[0]->count();
    Blob<Dtype> error_s;
    error_s.Reshape(N,C,H,W);
 subtrack_error<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,H,W,bottom[0]->gpu_data(),bottom[1]->gpu_data(),error_s.mutable_gpu_data());
cudaDeviceSynchronize();
  CUDA_POST_KERNEL_CHECK;


FindNotNaNs<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, error_s.gpu_data(), mask_.mutable_gpu_data());
  cudaDeviceSynchronize();
  CUDA_POST_KERNEL_CHECK;

 
   normalize_coeff_ =N*1*H*W;


// set masked (NaNs only) to zero
    KillMasked_acc<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, mask_.gpu_data(), error_s.mutable_gpu_data());
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;


   const int threads=N*1*H*W;

 compute_error<Dtype><<<CAFFE_GET_BLOCKS(threads), CAFFE_CUDA_NUM_THREADS>>>(
        threads, H,W,error_s.gpu_data(),error_.mutable_gpu_data());
  cudaDeviceSynchronize();
  CUDA_POST_KERNEL_CHECK;


   Dtype epe,acc;

   caffe_gpu_dot(error_.count(), error_.gpu_data(), sign_.gpu_data(), &epe);

   top[0]->mutable_cpu_data()[0]=epe/normalize_coeff_;


 
   caffe_gpu_set(sign_.count(),Dtype(0),sign_.mutable_gpu_data());
   Finderror<Dtype><<<CAFFE_GET_BLOCKS(threads), CAFFE_CUDA_NUM_THREADS>>>(threads,H,W,error_.gpu_data(),sign_.mutable_gpu_data(),2);
   caffe_gpu_dot(error_.count(), sign_.gpu_data(), sign_.gpu_data(), &acc);
   top[1]->mutable_cpu_data()[0]=acc/(float)threads;

   caffe_gpu_set(sign_.count(),Dtype(0),sign_.mutable_gpu_data());
   Finderror<Dtype><<<CAFFE_GET_BLOCKS(threads), CAFFE_CUDA_NUM_THREADS>>>(threads,H,W,error_.gpu_data(),sign_.mutable_gpu_data(),3);
   caffe_gpu_dot(error_.count(), sign_.gpu_data(), sign_.gpu_data(), &acc);
   top[1]->mutable_cpu_data()[1]=acc/(float)threads;

   caffe_gpu_set(sign_.count(),Dtype(0),sign_.mutable_gpu_data());
   Finderror<Dtype><<<CAFFE_GET_BLOCKS(threads), CAFFE_CUDA_NUM_THREADS>>>(threads,H,W,error_.gpu_data(),sign_.mutable_gpu_data(),4);
   caffe_gpu_dot(error_.count(), sign_.gpu_data(), sign_.gpu_data(), &acc);
   top[1]->mutable_cpu_data()[2]=acc/(float)threads;

   caffe_gpu_set(sign_.count(),Dtype(0),sign_.mutable_gpu_data());
   Finderror<Dtype><<<CAFFE_GET_BLOCKS(threads), CAFFE_CUDA_NUM_THREADS>>>(threads,H,W,error_.gpu_data(),sign_.mutable_gpu_data(),5);
   caffe_gpu_dot(error_.count(), sign_.gpu_data(), sign_.gpu_data(), &acc);
   top[1]->mutable_cpu_data()[3]=acc/(float)threads;


}



template<typename Dtype>
void FloaccLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

  CHECK(false) << "FloaccLayer cannot do backward.";   

}


INSTANTIATE_LAYER_GPU_FUNCS(FloaccLayer);

}	// namespace caffe
