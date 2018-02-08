#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/mean_layer.hpp"


namespace caffe {




template <typename Dtype>
__global__ void computemean(const int nthreads,int H,int W,int C,float scale, const Dtype* mean_,int mean_num, const Dtype* bottom, Dtype* top){
 CUDA_KERNEL_LOOP(index, nthreads) {
       // t=index% W;
      //  s=index/W %H;
       int j= index/(W*H) %C;
     //   i= index/(W*H*C);

    top[index]=scale*bottom[index]-mean_[j%mean_num];     

 }
}



template <typename Dtype>
void MeanLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

 for(int k=0; k<bottom.size();k++)
  {
      int N=bottom[k]->shape(0);
      int C=bottom[k]->shape(1);
      int H=bottom[k]->shape(2);
      int W=bottom[k]->shape(3);

      
     const int nthreads=N*C*W*H;
   
     computemean<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads,H,W,C,scale,mean_.gpu_data(),mean_num,
                                                                     bottom[k]->gpu_data(),top[k]->mutable_gpu_data());

  }




}




INSTANTIATE_LAYER_GPU_FUNCS(MeanLayer);

}	// namespace caffe
