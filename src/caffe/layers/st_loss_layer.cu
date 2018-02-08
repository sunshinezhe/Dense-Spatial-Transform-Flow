#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/st_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void STLossForwardGPU(const int nthreads, int N, int C,
		int output_H_, int output_W_, const Dtype* theta, Dtype* loss_array,Dtype* mask_) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

	        const Dtype* coordinates = theta + (output_H_ * output_W_ * C) * i;
		const int row_idx = output_W_ * s + t;

	  	const Dtype py = coordinates[row_idx ];
	  	const Dtype px = coordinates[row_idx +output_H_ * output_W_];

		
	  	const Dtype x = px + s;
	  	const Dtype y = py + t;
		
		Dtype loss = (Dtype)0;
		
		if(x < 0) {
			loss += (x ) * (x ) / 2.0;
		} else if(x > output_H_ - 1) {
			loss += (x -output_H_ + 1) * (x -output_H_ + 1) / 2.0;
		}
		
		if(y < 0) {
			loss += (y ) * (y) / 2;
		} else if(y > output_W_ - 1) {
			loss += (y - output_W_ +1) * (y - output_W_ +1) / 2.0;
		}
		if(loss>0.0000000001){
                    mask_[index]=1;}

		loss_array[index] = loss;
  }
}

template <typename Dtype>
void STLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	
	string prefix = "STLossLayer::Forward_gpu::\t";

	const Dtype* theta = bottom[0]->gpu_data();
	Dtype* loss_array = loss_.mutable_gpu_data();
	
	caffe_gpu_set(loss_.count(), (Dtype)0, loss_array);
 
        caffe_gpu_set(mask_.count(), (Dtype)0, mask_.mutable_gpu_data());
	
	const int nthreads = N * output_H_ * output_W_;
	STLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	     CAFFE_CUDA_NUM_THREADS>>>(nthreads, N,C, output_H_, output_W_, theta, loss_array,mask_.mutable_gpu_data());

        Dtype out_num;

        caffe_gpu_dot(mask_.count(), mask_.gpu_data(), mask_.gpu_data(), &out_num);
	
	Dtype loss;
	caffe_gpu_asum(nthreads, loss_array, &loss);
	
	if(out_num==0){
        top[0]->mutable_cpu_data()[0]=0;
        }else{
	top[0]->mutable_cpu_data()[0] = loss/out_num;
        }
        
}

template <typename Dtype>
__global__ void STLossBackwardGPU(const int nthreads, int N, int C,
		int output_H_, int output_W_, const Dtype* theta, Dtype* dtheta) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

	        const Dtype* coordinates = theta + (output_H_ * output_W_ * C) * i;
		const int row_idx = output_W_ * s + t;

	  	const Dtype py = coordinates[row_idx ];
	  	const Dtype px = coordinates[row_idx +output_H_ * output_W_];

		
	  	const Dtype x = px + s;
	  	const Dtype y = py + t;
		
		Dtype d1 = (Dtype)0, d2 = (Dtype)0;
		
		if(x < 0) {
			d1 = x;
		} else if(x > output_H_-1) {
			d1 = x -output_H_ + 1;
		}
		
		if(y < 0) {
			d2 = y ;
		} else if(y > output_W_ -1) {
			d2 = y-output_W_ +1;
		}
		
                int idx=i*(output_H_ * output_W_ * C) +output_W_ * s + t;
		dtheta[idx]=d2;
                dtheta[idx+output_H_ * output_W_]=d1;
  }
}

template <typename Dtype>
void STLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	const Dtype* theta = bottom[0]->gpu_data();
	
	
	Dtype* dtheta = bottom[0]->mutable_gpu_diff();
	
	
	const int nthreads = N * output_H_ * output_W_;
	STLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	     CAFFE_CUDA_NUM_THREADS>>>(nthreads, N,C, output_H_, output_W_, theta, dtheta);
	     
			
	caffe_gpu_scal(bottom[0]->count(), top[0]->cpu_diff()[0] , dtheta);
}

INSTANTIATE_LAYER_GPU_FUNCS(STLossLayer);

}  // namespace caffe
